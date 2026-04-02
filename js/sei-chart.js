(function () {
  'use strict';

  // ── Constants ──────────────────────────────────────────────────
  var ENVS = ['PushCube', 'PickCube', 'StackCube'];
  var ROBOTS = ['xArm6', 'Panda'];
  var INVALID_COMBOS = { 'StackCube|xArm6': true };

  var A_SOURCES = ['demos', 'rollout'];
  var A_USAGES = ['prefill', 'auxbc'];
  var B_TYPES = ['cql-h', 'cql-rho', 'calql', 'mcq', 'bc'];
  var C_TYPES = ['resrl', 'ibrl', 'cheq'];

  var COLOR_A = '#e53935';
  var COLOR_B = '#1e88e5';
  var COLOR_C = '#43a047';
  var COLOR_POS = '#4caf50';
  var COLOR_NEG = '#e53935';
  var COLOR_SAC = '#444';

  var TRANSITION_MS = 600;

  // ── State ──────────────────────────────────────────────────────
  var state = {
    envRobot: {},
    aSources: {},
    aUsages: {},
    bTypes: {},
    cTypes: {},
    keepAllVisible: false
  };

  function initState() {
    ENVS.forEach(function (env) {
      ROBOTS.forEach(function (robot) {
        if (!INVALID_COMBOS[env + '|' + robot]) {
          state.envRobot[env + '|' + robot] = true;
        }
      });
    });
    A_SOURCES.forEach(function (v) { state.aSources[v] = true; });
    A_USAGES.forEach(function (v) { state.aUsages[v] = true; });
    B_TYPES.forEach(function (v) { state.bTypes[v] = true; });
    C_TYPES.forEach(function (v) { state.cTypes[v] = true; });
  }

  // ── Data helpers ───────────────────────────────────────────────
  var rawData = [];

  function algoKey(d) {
    return [d.A_datasource, d.A_datausage, d.B_pretrainingtype, d.C_mixingtype].join('|');
  }

  function isSAC(d) {
    return d.A_datasource === 'none' && d.A_datausage === 'none' &&
           d.B_pretrainingtype === 'none' && d.C_mixingtype === 'none';
  }

  function passesStrategyFilter(d) {
    if (isSAC(d)) return true;

    if (d.A_datasource !== 'none') {
      if (!state.aSources[d.A_datasource]) return false;
    }
    if (d.A_datausage !== 'none') {
      var parts = d.A_datausage.split('-');
      for (var i = 0; i < parts.length; i++) {
        if (!state.aUsages[parts[i]]) return false;
      }
    }
    if (d.B_pretrainingtype !== 'none') {
      if (!state.bTypes[d.B_pretrainingtype]) return false;
    }
    if (d.C_mixingtype !== 'none') {
      if (!state.cTypes[d.C_mixingtype]) return false;
    }
    return true;
  }

  function computeDisplayData() {
    var activeKeys = Object.keys(state.envRobot).filter(function (k) { return state.envRobot[k]; });
    if (activeKeys.length === 0) return [];

    var activeSet = {};
    activeKeys.forEach(function (k) { activeSet[k] = true; });

    var groups = {};
    rawData.forEach(function (d) {
      var combo = d.env_id + '|' + d.robot;
      if (!activeSet[combo]) return;
      var key = algoKey(d);
      if (!groups[key]) {
        groups[key] = { sum: 0, count: 0, sample: d };
      }
      groups[key].sum += d.normalized_value;
      groups[key].count += 1;
    });

    var result = [];
    Object.keys(groups).forEach(function (key) {
      var g = groups[key];
      var d = g.sample;
      var passes = passesStrategyFilter(d);
      if (!state.keepAllVisible && !passes) return;
      result.push({
        key: key,
        A_datasource: d.A_datasource,
        A_datausage: d.A_datausage,
        B_pretrainingtype: d.B_pretrainingtype,
        C_mixingtype: d.C_mixingtype,
        sei: g.sum / g.count,
        highlighted: passes
      });
    });

    result.sort(function (a, b) { return b.sei - a.sei; });
    return result;
  }

  function computeSelectedAverage(data) {
    var selected = state.keepAllVisible
      ? data.filter(function (d) { return d.highlighted; })
      : data.slice();

    if (!selected.length) {
      return { value: 0, display: '0.00' };
    }

    var sum = selected.reduce(function (acc, d) { return acc + d.sei; }, 0);
    var avg = sum / selected.length;
    return { value: avg, display: avg.toFixed(2) };
  }

  function updateAverageBadge(avgInfo) {
    var badgeEl = d3.select('#sei-average-badge');
    var valueEl = d3.select('#sei-average-value');
    if (badgeEl.empty() || valueEl.empty()) return;

    badgeEl.classed('is-positive', avgInfo.value > 0);
    badgeEl.classed('is-negative', avgInfo.value < 0);

    var previous = parseFloat(valueEl.attr('data-current-value'));
    if (isNaN(previous)) previous = avgInfo.value;
    valueEl.attr('data-current-value', String(avgInfo.value));

    valueEl
      .interrupt()
      .style('opacity', 0.75)
      .transition()
      .duration(TRANSITION_MS)
      .ease(d3.easeCubicInOut)
      .styleTween('opacity', function () {
        return d3.interpolateNumber(0.75, 1);
      })
      .tween('text', function () {
        var i = d3.interpolateNumber(previous, avgInfo.value);
        return function (t) {
          this.textContent = i(t).toFixed(2);
        };
      });
  }

  // ── Label helpers ──────────────────────────────────────────────
  function buildLabelSpans(d) {
    if (isSAC(d)) return [{ text: 'SAC', color: COLOR_SAC }];

    var spans = [];
    var aParts = [];
    if (d.A_datasource !== 'none') aParts.push(d.A_datasource);
    if (d.A_datausage !== 'none') aParts.push(d.A_datausage);
    if (aParts.length) spans.push({ text: aParts.join('-'), color: COLOR_A });

    if (d.B_pretrainingtype !== 'none') {
      if (spans.length) spans.push({ text: ' + ', color: COLOR_SAC });
      spans.push({ text: d.B_pretrainingtype, color: COLOR_B });
    }
    if (d.C_mixingtype !== 'none') {
      if (spans.length) spans.push({ text: ' + ', color: COLOR_SAC });
      spans.push({ text: d.C_mixingtype, color: COLOR_C });
    }
    return spans;
  }

  // ── Control panel rendering ────────────────────────────────────
  function renderControls() {
    var root = document.getElementById('sei-controls');
    if (!root) return;
    root.innerHTML = '';

    // Section 1: Env / Robot
    var sec1 = el('div', 'sei-controls-section');
    sec1.appendChild(sectionTitle('Environment & Robot Selection'));
    var envSubtitle = el('div', 'sei-section-subtitle');
    envSubtitle.textContent = 'Tasks are ordered by difficulty from left to right';
    sec1.appendChild(envSubtitle);

    var grid1 = el('div', 'sei-env-robot-grid');
    ENVS.forEach(function (env) {
      var col = el('div', 'sei-env-col');

      var envBtn = toggleBtn(env, 'env-robot-btn');
      var anyActive = false;
      ROBOTS.forEach(function (r) {
        var k = env + '|' + r;
        if (!INVALID_COMBOS[k] && state.envRobot[k]) anyActive = true;
      });
      if (anyActive) envBtn.classList.add('active');
      envBtn.addEventListener('click', function () { toggleEnv(env); });
      col.appendChild(envBtn);

      var robotRow = el('div', 'sei-robot-row');
      ROBOTS.forEach(function (robot) {
        var k = env + '|' + robot;
        if (INVALID_COMBOS[k]) return;
        var rb = toggleBtn(robot, 'env-robot-btn');
        if (state.envRobot[k]) rb.classList.add('active');
        rb.setAttribute('data-combo', k);
        rb.addEventListener('click', function () { toggleCombo(k); });
        robotRow.appendChild(rb);
      });
      col.appendChild(robotRow);
      grid1.appendChild(col);
    });
    sec1.appendChild(grid1);
    root.appendChild(sec1);

    // Section 2: Algorithm Strategy
    var sec2 = el('div', 'sei-controls-section');
    sec2.appendChild(sectionTitle('Algorithm Strategy'));

    var toggleRow = el('div', 'sei-toggle-row');
    var toggleLabel = el('label', 'sei-toggle-label');
    toggleLabel.textContent = 'Keep all visible';
    var toggleSwitch = el('label', 'sei-toggle-switch');
    var toggleInput = document.createElement('input');
    toggleInput.type = 'checkbox';
    toggleInput.checked = state.keepAllVisible;
    toggleInput.addEventListener('change', function () {
      state.keepAllVisible = this.checked;
      renderControls();
      updateChart();
    });
    var toggleSlider = el('span', 'sei-toggle-slider');
    toggleSwitch.appendChild(toggleInput);
    toggleSwitch.appendChild(toggleSlider);
    toggleRow.appendChild(toggleLabel);
    toggleRow.appendChild(toggleSwitch);
    sec2.appendChild(toggleRow);

    var grid2 = el('div', 'sei-strategy-grid');

    grid2.appendChild(buildStrategyCol('A', 'strategy-a-btn', [
      { label: 'Source', items: A_SOURCES, stateObj: state.aSources },
      { label: 'Usage', items: A_USAGES, stateObj: state.aUsages }
    ]));
    grid2.appendChild(buildStrategyCol('B', 'strategy-b-btn', [
      { label: 'Pretraining', items: B_TYPES, stateObj: state.bTypes }
    ]));
    grid2.appendChild(buildStrategyCol('C', 'strategy-c-btn', [
      { label: 'Mixing', items: C_TYPES, stateObj: state.cTypes }
    ]));

    sec2.appendChild(grid2);
    root.appendChild(sec2);
  }

  function buildStrategyCol(name, btnClass, groups) {
    var col = el('div', 'sei-strategy-col');

    var allItems = [];
    groups.forEach(function (g) { allItems = allItems.concat(g.items); });

    var allActive = allItems.every(function (item) {
      for (var i = 0; i < groups.length; i++) {
        if (groups[i].stateObj[item] !== undefined) return groups[i].stateObj[item];
      }
      return false;
    });

    var master = toggleBtn('Strategy ' + name, btnClass + ' master-btn');
    if (allActive) master.classList.add('active');
    master.addEventListener('click', function () {
      var anyActive = allItems.some(function (item) {
        for (var i = 0; i < groups.length; i++) {
          if (groups[i].stateObj[item] !== undefined) return groups[i].stateObj[item];
        }
        return false;
      });
      var newVal = !anyActive;
      groups.forEach(function (g) {
        g.items.forEach(function (item) { g.stateObj[item] = newVal; });
      });
      renderControls();
      updateChart();
    });
    col.appendChild(master);

    groups.forEach(function (g) {
      var lbl = el('div', 'sei-variant-label');
      lbl.textContent = g.label;
      col.appendChild(lbl);
      var row = el('div', 'sei-variant-group');
      g.items.forEach(function (item) {
        var btn = toggleBtn(item, btnClass);
        if (g.stateObj[item]) btn.classList.add('active');
        btn.addEventListener('click', (function (stObj, it) {
          return function () {
            stObj[it] = !stObj[it];
            renderControls();
            updateChart();
          };
        })(g.stateObj, item));
        row.appendChild(btn);
      });
      col.appendChild(row);
    });

    return col;
  }

  function toggleEnv(env) {
    var combos = ROBOTS.map(function (r) { return env + '|' + r; })
      .filter(function (k) { return !INVALID_COMBOS[k]; });
    var anyActive = combos.some(function (k) { return state.envRobot[k]; });
    combos.forEach(function (k) { state.envRobot[k] = !anyActive; });
    renderControls();
    updateChart();
  }

  function toggleCombo(k) {
    state.envRobot[k] = !state.envRobot[k];
    renderControls();
    updateChart();
  }

  function el(tag, cls) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  function sectionTitle(text) {
    var d = el('div', 'sei-section-title');
    d.textContent = text;
    return d;
  }

  function toggleBtn(text, extraCls) {
    var b = el('button', 'toggle-btn' + (extraCls ? ' ' + extraCls : ''));
    b.type = 'button';
    b.textContent = text;
    return b;
  }

  // ── D3 chart ───────────────────────────────────────────────────
  // Left margin reserved for strategy text labels.
  var margin = { top: 30, right: 55, bottom: 40, left: 245 };
  // Visual thickness of each horizontal bar row (also drives chart height).
  var bandHeight = 16;
  var bandPad = 0.25;
  // Extra whitespace between the label divider line (x=0) and the plot area.
  var plotLeftPad = 45;
  var svgSel, gSel, xScale, yScale, xAxisG, yAxisG, zeroLineG;

  function initChart() {
    svgSel = d3.select('#sei-chart');
    gSel = svgSel.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    xScale = d3.scaleLinear().domain([-1, 1]);
    yScale = d3.scaleBand().padding(bandPad);

    xAxisG = gSel.append('g').attr('class', 'axis x-axis');
    yAxisG = gSel.append('g').attr('class', 'axis y-axis');
    zeroLineG = gSel.append('line').attr('class', 'zero-line');
    gSel.append('line').attr('class', 'label-separator');

    gSel.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .text('Sample Efficiency Improvement (SEI)');
  }

  function updateChart() {
    var data = computeDisplayData();
    var avgInfo = computeSelectedAverage(data);
    updateAverageBadge(avgInfo);
    var container = document.getElementById('sei-chart-container');
    if (!container) return;
    var totalWidth = container.clientWidth || 800;
    var innerW = totalWidth - margin.left - margin.right;
    var innerH = Math.max(data.length * bandHeight / (1 - bandPad), 60);
    var totalH = innerH + margin.top + margin.bottom;

    svgSel
      .attr('viewBox', '0 0 ' + totalWidth + ' ' + totalH)
      .attr('preserveAspectRatio', 'xMidYMin meet')
      .style('height', totalH + 'px');

    xScale.range([plotLeftPad, innerW]);
    yScale.domain(data.map(function (d) { return d.key; }))
      .range([0, innerH]);

    // x-axis
    xAxisG
      .attr('transform', 'translate(0,' + innerH + ')')
      .transition().duration(TRANSITION_MS)
      .call(d3.axisBottom(xScale).ticks(9));

    // axis label
    gSel.select('.axis-label')
      .attr('x', (plotLeftPad + innerW) / 2)
      .attr('y', innerH + margin.bottom - 4);

    // grid lines
    var gridVals = xScale.ticks(9);
    var gridLines = gSel.selectAll('.grid-line').data(gridVals, function (d) { return d; });
    gridLines.exit().remove();
    gridLines.enter().append('line').attr('class', 'grid-line')
      .merge(gridLines)
      .attr('x1', function (d) { return xScale(d); })
      .attr('x2', function (d) { return xScale(d); })
      .attr('y1', 0)
      .attr('y2', innerH);

    // zero line
    zeroLineG
      .attr('x1', xScale(0)).attr('x2', xScale(0))
      .attr('y1', 0).attr('y2', innerH);

    // label separator
    gSel.select('.label-separator')
      .attr('x1', 0).attr('x2', 0)
      .attr('y1', 0).attr('y2', innerH);

    // ── Data join ────────────────────────────────────────────────
    var bars = gSel.selectAll('.bar-group')
      .data(data, function (d) { return d.key; });

    // EXIT
    bars.exit()
      .transition().duration(TRANSITION_MS / 2)
      .style('opacity', 0)
      .remove();

    // ENTER
    var enter = bars.enter().append('g')
      .attr('class', 'bar-group')
      .attr('transform', function (d) {
        return 'translate(0,' + yScale(d.key) + ')';
      })
      .style('opacity', 0);

    enter.append('rect').attr('class', 'bar')
      .attr('height', yScale.bandwidth())
      .attr('rx', 3).attr('ry', 3)
      .attr('x', xScale(0))
      .attr('width', 0);

    enter.append('text').attr('class', 'bar-label')
      .attr('y', yScale.bandwidth() / 2)
      .attr('dy', '0.30em');

    enter.append('text').attr('class', 'bar-value')
      .attr('y', yScale.bandwidth() / 2)
      .attr('dy', '0.30em');

    // ENTER + UPDATE (merged)
    var merged = enter.merge(bars);

    merged.transition().duration(TRANSITION_MS)
      .ease(d3.easeCubicInOut)
      .style('opacity', function (d) {
        return (!state.keepAllVisible || d.highlighted) ? 1 : 0.25;
      })
      .style('filter', function (d) {
        return (state.keepAllVisible && !d.highlighted) ? 'saturate(0.8)' : null;
      })
      .attr('transform', function (d) {
        return 'translate(0,' + yScale(d.key) + ')';
      });

    merged.select('.bar')
      .transition().duration(TRANSITION_MS)
      .ease(d3.easeCubicInOut)
      .attr('height', yScale.bandwidth())
      .attr('x', function (d) { return d.sei >= 0 ? xScale(0) : xScale(d.sei); })
      .attr('width', function (d) { return Math.abs(xScale(d.sei) - xScale(0)); })
      .attr('fill', function (d) {
        if (d.sei === 0) return '#999';
        return d.sei > 0 ? COLOR_POS : COLOR_NEG;
      });

    // Multi-colored labels
    merged.each(function (d) {
      var labelEl = d3.select(this).select('.bar-label');
      var spans = buildLabelSpans(d);
      labelEl.selectAll('tspan').remove();
      labelEl.attr('x', -10).attr('text-anchor', 'end');
      spans.forEach(function (sp) {
        labelEl.append('tspan').text(sp.text).attr('fill', sp.color);
      });
      labelEl.transition().duration(TRANSITION_MS)
        .attr('y', yScale.bandwidth() / 2);
    });

    // Value annotations
    merged.select('.bar-value')
      .text(function (d) { return d.sei >= 0 ? '+' + d.sei.toFixed(3) : d.sei.toFixed(3); })
      .transition().duration(TRANSITION_MS)
      .attr('y', yScale.bandwidth() / 2)
      .attr('x', function (d) {
        if (d.sei >= 0) return xScale(d.sei) + 4;
        return xScale(d.sei) - 4;
      })
      .attr('text-anchor', function (d) { return d.sei >= 0 ? 'start' : 'end'; });
  }

  // ── Init ───────────────────────────────────────────────────────
  function init() {
    initState();
    renderControls();
    initChart();
    d3.json('data/sei_data.json').then(function (data) {
      rawData = data;
      updateChart();
    }).catch(function (err) {
      console.error('Failed to load SEI data:', err);
      var container = document.getElementById('sei-chart-container');
      if (container) {
        container.innerHTML = '<div class="notification is-warning">Could not load SEI data.</div>';
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
