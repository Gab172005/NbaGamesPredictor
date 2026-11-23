/* ==========================================================================
   NBA Game Outcome Predictor — Client-Side From-Scratch ML Inference & UI
   Mirrors data_utils.py and models.py in pure vanilla JavaScript.
   ========================================================================== */

(() => {
  "use strict";

  const FEATURE_LABELS = {
    home_games_played: "Home games played",
    visitor_games_played: "Away games played",
    home_win_pct: "Home season win %",
    visitor_win_pct: "Away season win %",
    diff_win_pct: "Season win % edge",
    home_last5_win_pct: "Home last-5 win %",
    visitor_last5_win_pct: "Away last-5 win %",
    diff_last5_win_pct: "Last-5 win % edge",
    home_avg_point_diff: "Home avg point margin",
    visitor_avg_point_diff: "Away avg point margin",
    diff_avg_point_diff: "Season point margin edge",
    home_last5_point_diff: "Home last-5 avg margin",
    visitor_last5_point_diff: "Away last-5 avg margin",
    diff_last5_point_diff: "Last-5 margin edge",
    home_avg_pts_scored: "Home avg points scored",
    visitor_avg_pts_scored: "Away avg points scored",
    home_avg_pts_allowed: "Home avg points allowed",
    visitor_avg_pts_allowed: "Away avg points allowed",
    home_rest_days: "Home rest days",
    visitor_rest_days: "Away rest days",
    diff_rest_days: "Rest days advantage",
    home_is_b2b: "Home on back-to-back",
    visitor_is_b2b: "Away on back-to-back",
  };

  const CHARTS = {
    final: {
      src: "./plots/final_model_comparison.png",
      alt: "Final NBA model comparison chart",
      caption:
        "Final showcase: All three tuned from-scratch models outperform the 58.29% Always-Pick-Home baseline across Accuracy, ROC-AUC, and 5-Fold CV.",
    },
    importance: {
      src: "./plots/feature_importance.png",
      alt: "Top 10 most important NBA stats",
      caption:
        "Top 10 stats learned by Tuned Logistic Regression: home point differential and road back-to-back fatigue carry the strongest signals.",
    },
    tuning: {
      src: "./plots/tuned_vs_untuned.png",
      alt: "Untuned vs tuned performance comparison",
      caption:
        "Hyperparameter tuning lifted Gaussian NB to #1 across Test Accuracy (63.64%), ROC-AUC (0.688), and Log-Loss (0.6188).",
    },
    roc: {
      src: "./plots/roc_curves.png",
      alt: "ROC curves on the 187-game chronological test set",
      caption:
        "ROC curves on the 187 unseen test games showing how clearly each model separates home wins from away wins above random chance (0.500).",
    },
    confusion: {
      src: "./plots/confusion_matrices.png",
      alt: "Confusion matrices for KNN, Gaussian NB, and Logistic Regression",
      caption:
        "Breakdown of correct picks vs. misses across the 109 actual home wins and 78 actual away wins in the test set.",
    },
    m3: {
      src: "./plots/model_metrics_comparison.png",
      alt: "Milestone 3 initial benchmark comparison",
      caption:
        "Milestone 3 baseline benchmark before hyperparameter tuning, comparing initial models against the Always-Pick-Home baseline.",
    },
  };

  // Application state
  let DATA = null;
  let TEAMS_BY_ABBR = {};
  let activeTestGame = null; // Holds exact test-game object when inspecting a real game
  let showAllBenchmarks = false;
  let explorerPage = 1;
  const PAGE_SIZE = 10;

  // Shortcut for getElementById
  const $ = (id) => document.getElementById(id);

  /* ------------------------------------------------------------------------
     1. Exact From-Scratch ML Math (mirrors data_utils.py & models.py)
     ------------------------------------------------------------------------ */

  function standardize(rawFeatures, scaler) {
    return rawFeatures.map((val, i) => (val - scaler.mean[i]) / scaler.std[i]);
  }

  function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
  }

  function sigmoid(z) {
    return 1.0 / (1.0 + Math.exp(-clamp(z, -500, 500)));
  }

  function predictGaussianNB(xScaled, model) {
    const logScore = (cls) => {
      const means = model.means[cls];
      const vars = model.vars[cls];
      let sum = Math.log(model.priors[cls]);
      for (let i = 0; i < xScaled.length; i++) {
        const diff = xScaled[i] - means[i];
        sum += -0.5 * (Math.log(2.0 * Math.PI * vars[i]) + (diff * diff) / vars[i]);
      }
      return sum;
    };
    const diff = logScore("1") - logScore("0");
    return sigmoid(diff);
  }

  function predictLogisticRegression(xScaled, model) {
    let z = model.bias;
    const contributions = [];
    for (let i = 0; i < xScaled.length; i++) {
      const contrib = xScaled[i] * model.weights[i];
      z += contrib;
      contributions.push({ index: i, value: contrib });
    }
    return { prob: sigmoid(z), contributions };
  }

  function predictKNN(xScaled, model) {
    const k = model.params.k;
    const XTrain = model.X_train;
    const yTrain = model.y_train;

    // Euclidean distance to each of the 744 historical training games
    const scored = new Array(XTrain.length);
    for (let rowIdx = 0; rowIdx < XTrain.length; rowIdx++) {
      const row = XTrain[rowIdx];
      let sumSq = 0;
      for (let col = 0; col < xScaled.length; col++) {
        const d = row[col] - xScaled[col];
        sumSq += d * d;
      }
      scored[rowIdx] = { idx: rowIdx, dist: Math.sqrt(sumSq) };
    }

    scored.sort((a, b) => a.dist - b.dist);
    const nearest = scored.slice(0, k);
    let homeWins = 0;
    for (const item of nearest) {
      homeWins += yTrain[item.idx];
    }

    return {
      prob: homeWins / k,
      homeWins,
      k,
      topNeighbors: nearest.slice(0, 5).map((item) => {
        const [date, matchup, score, homeWin] = model.train_games[item.idx].split("|");
        return {
          date,
          matchup,
          score,
          homeWin: Number(homeWin),
          dist: item.dist,
        };
      }),
    };
  }

  /* ------------------------------------------------------------------------
     2. Feature Vector Builder (matches exact 23-feature order in CSV)
     ------------------------------------------------------------------------ */

  function readFeatureVectorFromUI() {
    const homeGp = parseFloat($("home-gp").value) || 0;
    const awayGp = parseFloat($("away-gp").value) || 0;

    const homeWinPct = clamp((parseFloat($("home-win-pct").value) || 50) / 100, 0, 1);
    const awayWinPct = clamp((parseFloat($("away-win-pct").value) || 50) / 100, 0, 1);

    const homeLast5Win = clamp((parseFloat($("home-last5-win").value) || 50) / 100, 0, 1);
    const awayLast5Win = clamp((parseFloat($("away-last5-win").value) || 50) / 100, 0, 1);

    const homeScored = parseFloat($("home-pts-scored").value) || 113.5;
    const awayScored = parseFloat($("away-pts-scored").value) || 113.5;
    const homeAllowed = parseFloat($("home-pts-allowed").value) || 113.5;
    const awayAllowed = parseFloat($("away-pts-allowed").value) || 113.5;

    const homePointDiff = homeScored - homeAllowed;
    const awayPointDiff = awayScored - awayAllowed;

    const homeLast5Diff = parseFloat($("home-last5-diff").value) || 0;
    const awayLast5Diff = parseFloat($("away-last5-diff").value) || 0;

    const homeRest = parseFloat($("home-rest").value) || 2;
    const awayRest = parseFloat($("away-rest").value) || 2;

    const homeB2b = homeRest === 1 ? 1 : 0;
    const awayB2b = awayRest === 1 ? 1 : 0;

    return [
      homeGp,
      awayGp,
      homeWinPct,
      awayWinPct,
      homeWinPct - awayWinPct,
      homeLast5Win,
      awayLast5Win,
      homeLast5Win - awayLast5Win,
      homePointDiff,
      awayPointDiff,
      homePointDiff - awayPointDiff,
      homeLast5Diff,
      awayLast5Diff,
      homeLast5Diff - awayLast5Diff,
      homeScored,
      awayScored,
      homeAllowed,
      awayAllowed,
      homeRest,
      awayRest,
      homeRest - awayRest,
      homeB2b,
      awayB2b,
    ];
  }

  function populateStatInputsFromFeatures(features) {
    // Exact feature indices from data_utils.py
    $("home-gp").value = Math.round(features[0]);
    $("away-gp").value = Math.round(features[1]);
    $("home-win-pct").value = (features[2] * 100).toFixed(1);
    $("away-win-pct").value = (features[3] * 100).toFixed(1);
    $("home-last5-win").value = Math.round(features[5] * 100);
    $("away-last5-win").value = Math.round(features[6] * 100);
    $("home-last5-diff").value = features[11].toFixed(1);
    $("away-last5-diff").value = features[12].toFixed(1);
    $("home-pts-scored").value = features[14].toFixed(1);
    $("away-pts-scored").value = features[15].toFixed(1);
    $("home-pts-allowed").value = features[16].toFixed(1);
    $("away-pts-allowed").value = features[17].toFixed(1);

    const clampRest = (r) => String(Math.min(7, Math.max(1, Math.round(r))));
    $("home-rest").value = clampRest(features[18]);
    $("away-rest").value = clampRest(features[19]);
  }

  function populateStatInputsFromTeams(awayAbbr, homeAbbr) {
    const away = TEAMS_BY_ABBR[awayAbbr].stats;
    const home = TEAMS_BY_ABBR[homeAbbr].stats;

    $("away-gp").value = away.games_played;
    $("home-gp").value = home.games_played;
    $("away-win-pct").value = (away.win_pct * 100).toFixed(1);
    $("home-win-pct").value = (home.win_pct * 100).toFixed(1);
    $("away-last5-win").value = Math.round(away.last5_win_pct * 100);
    $("home-last5-win").value = Math.round(home.last5_win_pct * 100);
    $("away-pts-scored").value = away.avg_pts_scored.toFixed(1);
    $("home-pts-scored").value = home.avg_pts_scored.toFixed(1);
    $("away-pts-allowed").value = away.avg_pts_allowed.toFixed(1);
    $("home-pts-allowed").value = home.avg_pts_allowed.toFixed(1);
    $("away-last5-diff").value = away.last5_point_diff.toFixed(1);
    $("home-last5-diff").value = home.last5_point_diff.toFixed(1);
    $("away-rest").value = "2";
    $("home-rest").value = "2";
  }

  /* ------------------------------------------------------------------------
     3. Live Prediction Rendering
     ------------------------------------------------------------------------ */

  function fmtPct(prob) {
    return `${(prob * 100).toFixed(1)}%`;
  }

  function fmtSigned(num, decimals = 1) {
    return `${num >= 0 ? "+" : ""}${num.toFixed(decimals)}`;
  }

  function formatFeatureValue(name, value) {
    if (name.endsWith("_is_b2b")) return value >= 0.5 ? "Yes (B2B)" : "No";
    if (name.includes("win_pct")) {
      const pct = value * 100;
      return name.startsWith("diff_") ? `${fmtSigned(pct, 1)}%` : `${pct.toFixed(1)}%`;
    }
    if (name.includes("point_diff")) return `${fmtSigned(value, 1)} pts`;
    if (name.includes("pts_")) return `${value.toFixed(1)} PPG`;
    if (name.includes("rest_days")) {
      return name.startsWith("diff_") ? `${fmtSigned(value, 0)} d` : `${Math.round(value)} d`;
    }
    return String(Math.round(value));
  }

  function updatePrediction() {
    const awayAbbr = $("away-select").value;
    const homeAbbr = $("home-select").value;
    const awayTeam = TEAMS_BY_ABBR[awayAbbr];
    const homeTeam = TEAMS_BY_ABBR[homeAbbr];

    $("tape-away-name").textContent = `${awayTeam.abbr} · ${awayTeam.name}`;
    $("tape-home-name").textContent = `${homeTeam.name} · ${homeTeam.abbr}`;

    // Use exact feature row when inspecting an unmodified test game; otherwise read UI
    const rawFeatures = activeTestGame ? activeTestGame.features : readFeatureVectorFromUI();
    const xScaled = standardize(rawFeatures, DATA.scaler);

    // Update helper badges in Tale of the Tape
    const winDiffPct = (rawFeatures[2] - rawFeatures[3]) * 100;
    const winBadge = $("diff-win-pct-badge");
    if (Math.abs(winDiffPct) < 0.05) {
      winBadge.textContent = "Even";
      winBadge.className = "diff-chip diff-even";
    } else if (winDiffPct > 0) {
      winBadge.textContent = `${homeAbbr} +${winDiffPct.toFixed(1)}%`;
      winBadge.className = "diff-chip diff-home";
    } else {
      winBadge.textContent = `${awayAbbr} +${Math.abs(winDiffPct).toFixed(1)}%`;
      winBadge.className = "diff-chip diff-away";
    }

    const marginDiff = rawFeatures[8] - rawFeatures[9];
    const marginBadge = $("diff-point-diff-badge");
    if (Math.abs(marginDiff) < 0.05) {
      marginBadge.textContent = "Margin even";
      marginBadge.className = "diff-chip diff-even";
    } else if (marginDiff > 0) {
      marginBadge.textContent = `Margin: ${homeAbbr} +${marginDiff.toFixed(1)}`;
      marginBadge.className = "diff-chip diff-home";
    } else {
      marginBadge.textContent = `Margin: ${awayAbbr} +${Math.abs(marginDiff).toFixed(1)}`;
      marginBadge.className = "diff-chip diff-away";
    }

    $("away-b2b-badge").classList.toggle("hidden", rawFeatures[22] < 0.5);
    $("home-b2b-badge").classList.toggle("hidden", rawFeatures[21] < 0.5);

    // Run all 3 exact from-scratch models
    const gnbProb = predictGaussianNB(xScaled, DATA.models.gaussian_nb);
    const lrRes = predictLogisticRegression(xScaled, DATA.models.logistic_regression);
    const knnRes = predictKNN(xScaled, DATA.models.knn);

    // Primary Verdict (Tuned Gaussian NB)
    const homeFavored = gnbProb >= 0.5;
    const winnerTeam = homeFavored ? homeTeam : awayTeam;
    const winnerProb = homeFavored ? gnbProb : 1 - gnbProb;
    const sideLabel = homeFavored ? "Home" : "Away";

    $("verdict-headline").textContent = `${winnerTeam.name} (${sideLabel}) favored to win`;
    $("verdict-subtext").textContent = `Estimated ${fmtPct(winnerProb)} win probability (${awayTeam.abbr} @ ${homeTeam.abbr}).`;

    // Split bar
    const awayPct = ((1 - gnbProb) * 100).toFixed(1);
    const homePct = (gnbProb * 100).toFixed(1);
    $("away-bar-label").textContent = `${awayTeam.abbr} (Away) ${awayPct}%`;
    $("home-bar-label").textContent = `${homeTeam.abbr} (Home) ${homePct}%`;
    $("prob-bar-away").style.width = `${awayPct}%`;
    $("prob-bar-home").style.width = `${homePct}%`;

    // Consensus pill
    const homeVotes = [gnbProb >= 0.5, lrRes.prob >= 0.5, knnRes.prob >= 0.5].filter(Boolean).length;
    const consensusBadge = $("consensus-badge");
    if (homeVotes === 3 || homeVotes === 0) {
      const teamPick = homeVotes === 3 ? homeTeam.abbr : awayTeam.abbr;
      consensusBadge.textContent = `3/3 models pick ${teamPick}`;
      consensusBadge.className = "consensus-badge consensus-unanimous";
    } else {
      const majTeam = homeVotes >= 2 ? homeTeam.abbr : awayTeam.abbr;
      consensusBadge.textContent = `2/3 models pick ${majTeam}`;
      consensusBadge.className = "consensus-badge consensus-split";
    }

    // Individual model cards
    renderModelCard("gnb", gnbProb, awayAbbr, homeAbbr);
    renderModelCard("lr", lrRes.prob, awayAbbr, homeAbbr);
    renderModelCard("knn", knnRes.prob, awayAbbr, homeAbbr);
    $("knn-note").textContent = `${knnRes.homeWins} of the ${knnRes.k} most similar past games were won by the home team.`;

    // Actual outcome banner if inspecting an unseen holdout test game
    const actualBanner = $("actual-outcome-banner");
    if (activeTestGame) {
      const actualWinner = activeTestGame.homeWin === 1 ? activeTestGame.home : activeTestGame.visitor;
      const gnbCorrect = (gnbProb >= 0.5 ? 1 : 0) === activeTestGame.homeWin;
      actualBanner.innerHTML = `
        <span>
          <strong>Real Test Game (${activeTestGame.date}):</strong>
          ${activeTestGame.visitor} ${activeTestGame.visitorPts} – ${activeTestGame.homePts} ${activeTestGame.home}
          &nbsp;·&nbsp; Actual Winner: <strong>${actualWinner}</strong>
        </span>
        <span class="${gnbCorrect ? "mark-ok" : "mark-miss"}">
          ${gnbCorrect ? "✓ Primary model predicted correctly" : "✕ Primary model missed (upset)"}
        </span>
      `;
      actualBanner.classList.remove("hidden");
    } else {
      actualBanner.classList.add("hidden");
    }

    // Explainability: top 5 Logistic Regression drivers
    renderDrivers(lrRes.contributions, rawFeatures, awayAbbr, homeAbbr);

    // Explainability: top 5 closest KNN training games
    renderNeighbors(knnRes.topNeighbors);
  }

  function renderModelCard(prefix, prob, awayAbbr, homeAbbr) {
    const picksHome = prob >= 0.5;
    const pickEl = $(`${prefix}-pick`);
    pickEl.textContent = picksHome ? `Pick: ${homeAbbr}` : `Pick: ${awayAbbr}`;
    pickEl.className = `pick-pill ${picksHome ? "pick-home" : "pick-away"}`;

    $(`${prefix}-prob`).textContent = fmtPct(prob);
    const bar = $(`${prefix}-bar`);
    bar.style.width = `${(prob * 100).toFixed(1)}%`;
    bar.className = `mini-fill ${picksHome ? "" : "is-away"}`;
  }

  function renderDrivers(contributions, rawFeatures, awayAbbr, homeAbbr) {
    const featureNames = DATA.metadata.featureNames;
    const sorted = [...contributions]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5);
    const maxAbs = Math.max(...sorted.map((d) => Math.abs(d.value)), 0.01);

    const container = $("drivers-list");
    container.innerHTML = sorted
      .map((item) => {
        const name = featureNames[item.index];
        const label = FEATURE_LABELS[name] || name;
        const formattedVal = formatFeatureValue(name, rawFeatures[item.index]);
        const favorsHome = item.value >= 0;
        const widthPct = Math.max(8, (Math.abs(item.value) / maxAbs) * 100).toFixed(1);
        const sideText = favorsHome ? `Favors ${homeAbbr}` : `Favors ${awayAbbr}`;
        return `
          <div class="driver-row">
            <div class="driver-top">
              <span class="driver-name">${label} <span style="color:#64748b;font-weight:500;">(${formattedVal})</span></span>
              <span class="driver-impact ${favorsHome ? "text-home" : "text-away"}">${sideText}</span>
            </div>
            <div class="driver-bar-track">
              <div class="driver-bar-fill ${favorsHome ? "driver-bar-home" : "driver-bar-away"}" style="width:${widthPct}%"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  function renderNeighbors(neighbors) {
    const container = $("neighbors-list");
    container.innerHTML = neighbors
      .map((n) => {
        const homeWon = n.homeWin === 1;
        return `
          <div class="neighbor-row">
            <div class="neighbor-meta">
              <span class="neighbor-match">${n.matchup} <span style="color:#64748b;font-weight:500;">(${n.score})</span></span>
              <span class="neighbor-date">${n.date} · Similarity distance: ${n.dist.toFixed(2)}</span>
            </div>
            <span class="outcome-chip ${homeWon ? "chip-home" : "chip-away"}">
              ${homeWon ? "Home Win" : "Away Win"}
            </span>
          </div>
        `;
      })
      .join("");
  }

  /* ------------------------------------------------------------------------
     4. Unseen Test Games Explorer
     ------------------------------------------------------------------------ */

  function getFilteredTestGames() {
    const team = $("filter-team").value;
    const outcome = $("filter-outcome").value;

    // Show newest games first
    return [...DATA.testGames].reverse().filter((g) => {
      if (team !== "ALL" && g.home !== team && g.visitor !== team) return false;
      const gnbRight = (g.probs.gnb >= 0.5 ? 1 : 0) === g.homeWin;
      if (outcome === "GNB_RIGHT" && !gnbRight) return false;
      if (outcome === "GNB_WRONG" && gnbRight) return false;
      if (outcome === "HOME_WIN" && g.homeWin !== 1) return false;
      if (outcome === "AWAY_WIN" && g.homeWin !== 0) return false;
      return true;
    });
  }

  function renderTestGamesTable() {
    const filtered = getFilteredTestGames();
    const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    explorerPage = clamp(explorerPage, 1, totalPages);

    const gnbHits = filtered.filter((g) => (g.probs.gnb >= 0.5 ? 1 : 0) === g.homeWin).length;
    const lrHits = filtered.filter((g) => (g.probs.lr >= 0.5 ? 1 : 0) === g.homeWin).length;
    const knnHits = filtered.filter((g) => (g.probs.knn >= 0.5 ? 1 : 0) === g.homeWin).length;

    const pct = (hits) => (filtered.length ? `${((hits / filtered.length) * 100).toFixed(1)}%` : "0.0%");
    $("explorer-summary").textContent =
      `${filtered.length} games · GNB ${pct(gnbHits)} · LR ${pct(lrHits)} · KNN ${pct(knnHits)}`;

    const start = (explorerPage - 1) * PAGE_SIZE;
    const slice = filtered.slice(start, start + PAGE_SIZE);

    const fmtCell = (prob, game) => {
      const picksHome = prob >= 0.5;
      const pick = picksHome ? game.home : game.visitor;
      const ok = (picksHome ? 1 : 0) === game.homeWin;
      const conf = picksHome ? prob : 1 - prob;
      return `
        <span class="pred-cell">
          <span class="${ok ? "mark-ok" : "mark-miss"}">${ok ? "✓" : "✕"}</span>
          ${pick} <span style="color:#64748b;font-weight:400;">(${(conf * 100).toFixed(0)}%)</span>
        </span>
      `;
    };

    const tbody = $("test-games-tbody");
    if (slice.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#64748b;padding:24px;">No test games match this filter.</td></tr>`;
    } else {
      tbody.innerHTML = slice
        .map((g) => {
          const actualWinner = g.homeWin === 1 ? g.home : g.visitor;
          return `
            <tr>
              <td>${g.date}</td>
              <td>
                <strong>${g.visitor}</strong> ${g.visitorPts} @ <strong>${g.home}</strong> ${g.homePts}
              </td>
              <td>
                <span class="outcome-chip ${g.homeWin === 1 ? "chip-home" : "chip-away"}">
                  ${actualWinner}
                </span>
              </td>
              <td>${fmtCell(g.probs.gnb, g)}</td>
              <td>${fmtCell(g.probs.lr, g)}</td>
              <td>${fmtCell(g.probs.knn, g)}</td>
              <td class="num">
                <button type="button" class="inspect-btn" data-game-id="${g.id}">Inspect</button>
              </td>
            </tr>
          `;
        })
        .join("");
    }

    $("page-info").textContent = `Page ${explorerPage} of ${totalPages} (${filtered.length} games)`;
    $("prev-page-btn").disabled = explorerPage <= 1;
    $("next-page-btn").disabled = explorerPage >= totalPages;
  }

  function loadTestGameById(gameId, shouldScroll = false) {
    const game = DATA.testGames.find((g) => g.id === gameId);
    if (!game) return;

    activeTestGame = game;
    $("away-select").value = game.visitor;
    $("home-select").value = game.home;
    $("test-game-select").value = game.id;
    populateStatInputsFromFeatures(game.features);
    updatePrediction();

    if (shouldScroll) {
      $("predictor").scrollIntoView({ behavior: "smooth" });
    }
  }

  /* ------------------------------------------------------------------------
     5. Benchmarks Table & Chart Switcher
     ------------------------------------------------------------------------ */

  function renderBenchmarksTable() {
    const rows = showAllBenchmarks
      ? DATA.benchmarks
      : DATA.benchmarks.filter((r) => r.variant === "Baseline" || r.variant === "Tuned");

    const bestAcc = Math.max(...rows.map((r) => r.accuracy));
    const bestF1 = Math.max(...rows.map((r) => r.f1));
    const bestAuc = Math.max(...rows.map((r) => r.roc_auc));
    const bestLoss = Math.min(...rows.map((r) => r.log_loss));
    const bestCv = Math.max(...rows.map((r) => r.cv_acc_mean));

    const formatParams = (raw) => {
      if (raw === "home_win=1") return "Always Home";
      try {
        const obj = JSON.parse(raw);
        return Object.entries(obj)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ");
      } catch {
        return raw;
      }
    };

    const wrapBest = (text, isBest) => (isBest ? `<span class="best-metric">${text}</span>` : text);

    $("benchmarks-tbody").innerHTML = rows
      .map((r) => {
        const isHighlight = r.label === "Gaussian NB (Tuned)";
        return `
          <tr class="${isHighlight ? "row-highlight" : ""}">
            <td><strong>${r.label}</strong></td>
            <td><code>${formatParams(r.params)}</code></td>
            <td class="num">${wrapBest(`${(r.accuracy * 100).toFixed(2)}%`, r.accuracy === bestAcc)}</td>
            <td class="num">${wrapBest(`${(r.f1 * 100).toFixed(2)}%`, r.f1 === bestF1)}</td>
            <td class="num">${wrapBest(r.roc_auc.toFixed(3), r.roc_auc === bestAuc)}</td>
            <td class="num">${wrapBest(r.log_loss.toFixed(4), r.log_loss === bestLoss)}</td>
            <td class="num">${wrapBest(`${(r.cv_acc_mean * 100).toFixed(2)}%`, r.cv_acc_mean === bestCv)}</td>
          </tr>
        `;
      })
      .join("");
  }

  function selectChart(key) {
    const chart = CHARTS[key];
    if (!chart) return;
    $("chart-image").src = chart.src;
    $("chart-image").alt = chart.alt;
    $("chart-caption").textContent = chart.caption;

    document.querySelectorAll(".chart-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.chart === key);
    });
  }

  /* ------------------------------------------------------------------------
     6. Initialization & Event Listeners
     ------------------------------------------------------------------------ */

  function populateDropdowns() {
    const awaySelect = $("away-select");
    const homeSelect = $("home-select");
    const filterTeam = $("filter-team");

    const teamOptions = DATA.teams
      .map((t) => `<option value="${t.abbr}">${t.abbr} — ${t.name} (${(t.stats.win_pct * 100).toFixed(0)}% W)</option>`)
      .join("");

    awaySelect.innerHTML = teamOptions;
    homeSelect.innerHTML = teamOptions;

    filterTeam.innerHTML =
      `<option value="ALL">All 30 Teams</option>` +
      DATA.teams.map((t) => `<option value="${t.abbr}">${t.abbr} — ${t.name}</option>`).join("");

    // Default compelling matchup: Denver Nuggets @ Boston Celtics
    awaySelect.value = TEAMS_BY_ABBR.DEN ? "DEN" : DATA.teams[0].abbr;
    homeSelect.value = TEAMS_BY_ABBR.BOS ? "BOS" : DATA.teams[1].abbr;

    // Populate unseen test game quick-loader (newest first)
    const testSelect = $("test-game-select");
    const gameOptions = [...DATA.testGames]
      .reverse()
      .map((g) => {
        const winner = g.homeWin === 1 ? g.home : g.visitor;
        return `<option value="${g.id}">${g.date} | ${g.visitor} @ ${g.home} (${g.visitorPts}–${g.homePts}, ${winner} won)</option>`;
      })
      .join("");
    testSelect.innerHTML =
      `<option value="">Custom team matchup (latest 2022–23 stats)</option>` + gameOptions;
  }

  function wireEvents() {
    const onTeamSelectionChange = () => {
      activeTestGame = null;
      $("test-game-select").value = "";
      populateStatInputsFromTeams($("away-select").value, $("home-select").value);
      updatePrediction();
    };

    $("away-select").addEventListener("change", onTeamSelectionChange);
    $("home-select").addEventListener("change", onTeamSelectionChange);

    $("swap-teams-btn").addEventListener("click", () => {
      const prevAway = $("away-select").value;
      $("away-select").value = $("home-select").value;
      $("home-select").value = prevAway;
      onTeamSelectionChange();
    });

    $("reset-stats-btn").addEventListener("click", onTeamSelectionChange);

    $("test-game-select").addEventListener("change", (e) => {
      if (!e.target.value) {
        onTeamSelectionChange();
      } else {
        loadTestGameById(e.target.value, false);
      }
    });

    $("random-game-btn").addEventListener("click", () => {
      const idx = Math.floor(Math.random() * DATA.testGames.length);
      loadTestGameById(DATA.testGames[idx].id, false);
    });

    // Any manual edit switches smoothly into live custom mode
    const statInputIds = [
      "away-win-pct", "home-win-pct",
      "away-pts-scored", "home-pts-scored",
      "away-pts-allowed", "home-pts-allowed",
      "away-last5-win", "home-last5-win",
      "away-last5-diff", "home-last5-diff",
      "away-rest", "home-rest",
      "away-gp", "home-gp",
    ];
    statInputIds.forEach((id) => {
      const el = $(id);
      const evt = el.tagName === "SELECT" ? "change" : "input";
      el.addEventListener(evt, () => {
        activeTestGame = null;
        $("test-game-select").value = "";
        updatePrediction();
      });
    });

    // Explorer filters & inspect clicks
    $("filter-team").addEventListener("change", () => {
      explorerPage = 1;
      renderTestGamesTable();
    });
    $("filter-outcome").addEventListener("change", () => {
      explorerPage = 1;
      renderTestGamesTable();
    });
    $("prev-page-btn").addEventListener("click", () => {
      explorerPage -= 1;
      renderTestGamesTable();
    });
    $("next-page-btn").addEventListener("click", () => {
      explorerPage += 1;
      renderTestGamesTable();
    });
    $("test-games-tbody").addEventListener("click", (e) => {
      const btn = e.target.closest(".inspect-btn");
      if (btn) {
        loadTestGameById(btn.dataset.gameId, true);
      }
    });

    // Benchmark table toggle
    $("toggle-tuned").addEventListener("click", () => {
      showAllBenchmarks = false;
      $("toggle-tuned").classList.add("active");
      $("toggle-all").classList.remove("active");
      renderBenchmarksTable();
    });
    $("toggle-all").addEventListener("click", () => {
      showAllBenchmarks = true;
      $("toggle-all").classList.add("active");
      $("toggle-tuned").classList.remove("active");
      renderBenchmarksTable();
    });

    // Charts gallery & modal zoom
    document.querySelectorAll(".chart-tab").forEach((btn) => {
      btn.addEventListener("click", () => selectChart(btn.dataset.chart));
    });

    const modal = $("chart-modal");
    $("chart-image").addEventListener("click", () => {
      $("modal-image").src = $("chart-image").src;
      $("modal-image").alt = $("chart-image").alt;
      modal.classList.remove("hidden");
    });
    $("modal-close-btn").addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") modal.classList.add("hidden");
    });
  }

  async function init() {
    try {
      const response = await fetch("./frontend/model_data.json");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} while loading model_data.json`);
      }
      DATA = await response.json();
      TEAMS_BY_ABBR = Object.fromEntries(DATA.teams.map((t) => [t.abbr, t]));

      populateDropdowns();
      wireEvents();
      populateStatInputsFromTeams($("away-select").value, $("home-select").value);
      updatePrediction();
      renderTestGamesTable();
      renderBenchmarksTable();
      selectChart("final");
    } catch (err) {
      const banner = $("status-banner");
      banner.innerHTML =
        `<strong>Could not load model data.</strong> If you opened <code>index.html</code> directly as a local file, start a quick local server with <code>python -m http.server 8000</code> and visit <code>http://localhost:8000</code> (or deploy to GitHub Pages).`;
      banner.classList.remove("hidden");
      console.error(err);
    }
  }

  init();
})();
