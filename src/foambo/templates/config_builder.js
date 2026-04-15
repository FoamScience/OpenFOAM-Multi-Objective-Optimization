/**
 * foamBO Config Builder — Alpine.js application
 *
 * Manages form state, YAML preview, export/load, and preflight/dry-run actions.
 */

function configBuilder() {
  return {
    activeSection: 'experiment',
    copied: false,
    preflightRunning: false,
    dryRunRunning: false,
    preflightResults: null,
    validationErrors: [],
    showConcepts: false,
    docsFilter: 'all',
    docsLoading: false,
    docsEntries: [],
    fieldHints: {},

    bootstrapPreview: null,
    bootstrapLoading: false,
    bootstrapError: '',

    sections: [
      { id: 'bootstrap', label: 'Bootstrap' },
      { id: 'experiment', label: 'Experiment' },
      { id: 'trial_generation', label: 'Trial Generation' },
      { id: 'baseline', label: 'Baseline' },
      { id: 'optimization', label: 'Optimization' },
      { id: 'orchestration', label: 'Orchestration' },
      { id: 'store', label: 'Store' },
      {
        id: 'advanced', label: 'Advanced',
        children: [
          { id: 'advanced.dimred', label: 'Dimensionality Reduction' },
          { id: 'advanced.deps', label: 'Trial Dependencies' },
        ]
      },
    ],

    config: {
      version: '1.2.0',
      bootstrap: null,
      specialize: null,
      experiment: {
        name: '',
        description: '',
        parameters: [],
        parameter_constraints: [],
      },
      trial_generation: {
        method: 'fast',
        initialization_budget: null,
        initialize_with_center: true,
        generation_nodes: [],
      },
      baseline: {
        parameters: {},
      },
      optimization: {
        objective: '',
        outcome_constraints: [],
        objective_thresholds: [],
        metrics: [],
        case_runner: {
          template_case: '',
          mode: 'local',
          runner: '',
          trial_destination: '',
          artifacts_folder: './artifacts',
          remote_status_query: '',
          remote_early_stop: '',
          variable_substitution: [],
          file_substitution: [],
        },
      },
      orchestration_settings: {
        max_trials: 50,
        parallelism: 1,
        tolerated_trial_failure_rate: 0.5,
        timeout_hours: 24,
        ttl_seconds_for_trials: 600,
        initial_seconds_between_polls: 10,
        seconds_between_polls_backoff_factor: 1.0,
        global_stopping_strategy: {
          min_trials: 20,
          window_size: 10,
          improvement_bar: 0.1,
        },
        early_stopping_strategy: {
          type: '',
        },
        dimensionality_reduction: {
          enabled: false,
          after_trials: 10,
          min_importance: 0.05,
          fix_at: 'best',
          max_fix_fraction: 0.5,
        },
      },
      store: {
        save_to: 'json',
        read_from: 'nowhere',
        backend_options: { url: null },
      },
      trial_dependencies: [],
    },

    init() {
      // Load field hints for tooltips
      this.loadFieldHints();
      // Re-apply tooltips when switching sections (new labels become visible)
      this.$watch('activeSection', () => this.$nextTick(() => this.applyTooltips()));
    },

    /** Render an early stopping node as HTML. Recursive for and/or composites. */
    renderEsNode(node, path) {
      if (!node) return '';
      const typeSelect = `
        <div class="form-group">
          <label>Type</label>
          <select @change="
            let n = ${path};
            const t = $event.target.value;
            if (t === '') { Object.keys(n).forEach(k => delete n[k]); n.type = ''; }
            else if (t === 'and' || t === 'or') {
              Object.keys(n).forEach(k => delete n[k]);
              n.type = t;
              n.left = { type: 'percentile', metric_names: [], percentile_threshold: 25, min_progression: 2 };
              n.right = { type: 'percentile', metric_names: [], percentile_threshold: 25, min_progression: 2 };
            } else {
              Object.keys(n).forEach(k => delete n[k]);
              n.type = t;
              n.metric_names = [];
              if (t === 'percentile') { n.percentile_threshold = 25; }
              if (t === 'threshold') { n.threshold = 0; }
              n.min_progression = 2;
            }
          ">
            <option value="" ${node.type === '' ? 'selected' : ''}>None</option>
            <option value="percentile" ${node.type === 'percentile' ? 'selected' : ''}>percentile</option>
            <option value="threshold" ${node.type === 'threshold' ? 'selected' : ''}>threshold</option>
            <option value="and" ${node.type === 'and' ? 'selected' : ''}>and (composite)</option>
            <option value="or" ${node.type === 'or' ? 'selected' : ''}>or (composite)</option>
          </select>
        </div>`;

      if (!node.type) return typeSelect;

      if (node.type === 'percentile' || node.type === 'threshold') {
        const selected = new Set(node.metric_names || []);
        const metricOptions = (this.config.optimization.metrics || [])
          .map(m => m.name).filter(Boolean);
        let metricPicker;
        if (metricOptions.length === 0) {
          metricPicker = `<small class="hint">No metrics defined yet — add metrics in the Optimization section.</small>`;
        } else {
          metricPicker = metricOptions.map(name => `
            <label class="toggle-label" style="display:inline-flex;margin-right:0.75rem">
              <input type="checkbox" ${selected.has(name) ? 'checked' : ''}
                @change="
                  const arr = ${path}.metric_names || [];
                  if ($event.target.checked) { if (!arr.includes('${this._esc(name)}')) arr.push('${this._esc(name)}'); }
                  else { const i = arr.indexOf('${this._esc(name)}'); if (i >= 0) arr.splice(i, 1); }
                  ${path}.metric_names = arr;
                ">
              ${this._esc(name)}
            </label>`).join('');
        }
        let fields = `
          <div class="form-group" style="grid-column:1/-1">
            <label>Metric names</label>
            <div>${metricPicker}</div>
          </div>
          <div class="param-grid">`;
        if (node.type === 'percentile') {
          fields += `
            <div class="form-group">
              <label>Percentile threshold</label>
              <input type="number" value="${node.percentile_threshold ?? 25}" min="0" max="100"
                @change="${path}.percentile_threshold = Number($event.target.value)">
            </div>`;
        }
        if (node.type === 'threshold') {
          fields += `
            <div class="form-group">
              <label>Threshold</label>
              <input type="number" value="${node.threshold ?? 0}" step="any"
                @change="${path}.threshold = Number($event.target.value)">
            </div>`;
        }
        fields += `
            <div class="form-group">
              <label>Min progression</label>
              <input type="number" value="${node.min_progression ?? 2}" min="0"
                @change="${path}.min_progression = Number($event.target.value)">
            </div>
          </div>`;
        return typeSelect + fields;
      }

      if (node.type === 'and' || node.type === 'or') {
        // Ensure left/right exist
        if (!node.left) node.left = { type: '' };
        if (!node.right) node.right = { type: '' };
        return typeSelect + `
          <div class="es-composite">
            <div class="es-branch">
              <label class="es-branch-label">Left</label>
              <div class="es-branch-content" x-html="renderEsNode(${path}.left, '${path}.left')"></div>
            </div>
            <div class="es-branch">
              <label class="es-branch-label">Right</label>
              <div class="es-branch-content" x-html="renderEsNode(${path}.right, '${path}.right')"></div>
            </div>
          </div>`;
      }

      return typeSelect;
    },

    _esc(s) { return String(s).replace(/"/g, '&quot;').replace(/</g, '&lt;'); },

    _importEsNode(data) {
      if (!data || !data.type) return { type: '' };
      const node = { type: data.type };
      if (data.type === 'and' || data.type === 'or') {
        node.left = this._importEsNode(data.left);
        node.right = this._importEsNode(data.right);
      } else {
        node.metric_names = data.metric_names || data.metric_signatures || [];
        if (data.type === 'percentile') node.percentile_threshold = data.percentile_threshold ?? 25;
        if (data.type === 'threshold') node.threshold = data.threshold ?? 0;
        node.min_progression = data.min_progression ?? 2;
      }
      return node;
    },

    get yamlOutput() {
      return jsyaml.dump(this.buildOutput(), {
        indent: 2,
        lineWidth: 120,
        noRefs: true,
        sortKeys: false,
        skipInvalid: true,
      });
    },

    /** Build clean output object, omitting empty/default values. */
    buildOutput() {
      const cfg = JSON.parse(JSON.stringify(this.config));
      const out = { version: cfg.version };

      // Bootstrap (emit only if a path is set)
      if (cfg.bootstrap) {
        out.bootstrap = cfg.bootstrap;
        if (cfg.specialize && Object.keys(cfg.specialize).length) {
          out.specialize = cfg.specialize;
        }
      }

      // Experiment
      const exp = { name: cfg.experiment.name };
      if (cfg.experiment.description) exp.description = cfg.experiment.description;
      exp.parameters = cfg.experiment.parameters.map(p => {
        const param = { name: p.name, parameter_type: p.parameter_type };
        if (p.kind === 'choice') {
          param.values = p.values || [];
        } else {
          param.bounds = [p.bounds_lower, p.bounds_upper];
          if (p.scaling) param.scaling = p.scaling;
          if (p.step_size) param.step_size = p.step_size;
        }
        if (p.groups?.length) param.groups = p.groups;
        return param;
      });
      if (cfg.experiment.parameter_constraints?.length) {
        exp.parameter_constraints = cfg.experiment.parameter_constraints;
      }
      out.experiment = exp;

      // Trial generation
      const tg = { method: cfg.trial_generation.method };
      if (cfg.trial_generation.method === 'custom' && cfg.trial_generation.generation_nodes?.length) {
        tg.generation_nodes = cfg.trial_generation.generation_nodes.map(n => {
          const node = { node_name: n.node_name, generator_specs: [] };
          node.generator_specs = (n.generator_specs || []).map(s => {
            const spec = { generator_enum: s.generator_enum };
            if (s.model_kwargs) spec.model_kwargs = s.model_kwargs;
            return spec;
          });
          if (n.transition_criteria?.length) {
            node.transition_criteria = n.transition_criteria.map(tc => {
              const out = {};
              out[tc.type] = tc.type === 'auto_transition_after_gen' ? true : tc.value;
              return out;
            });
          }
          return node;
        });
      }
      if (cfg.trial_generation.initialization_budget != null && cfg.trial_generation.method !== 'custom') {
        tg.initialization_budget = cfg.trial_generation.initialization_budget;
      }
      if (!cfg.trial_generation.initialize_with_center) tg.initialize_with_center = false;
      out.trial_generation = tg;

      // Baseline
      const blKeys = Object.keys(cfg.baseline.parameters).filter(k => cfg.baseline.parameters[k] !== '' && cfg.baseline.parameters[k] != null);
      if (blKeys.length) {
        const bl = {};
        blKeys.forEach(k => { bl[k] = cfg.baseline.parameters[k]; });
        out.baseline = { parameters: bl };
      }

      // Optimization
      const opt = {};
      if (cfg.optimization.objective) opt.objective = cfg.optimization.objective;
      if (cfg.optimization.outcome_constraints?.length) opt.outcome_constraints = cfg.optimization.outcome_constraints;
      if (cfg.optimization.objective_thresholds?.length) opt.objective_thresholds = cfg.optimization.objective_thresholds;
      if (cfg.optimization.metrics?.length) {
        opt.metrics = cfg.optimization.metrics.map(m => {
          const metric = { name: m.name, command: m.command };
          if (m.progress?.length) metric.progress = m.progress;
          if (m.lower_is_better) metric.lower_is_better = true;
          return metric;
        });
      }
      // Case runner
      const cr = {};
      if (cfg.optimization.case_runner.template_case) cr.template_case = cfg.optimization.case_runner.template_case;
      cr.mode = cfg.optimization.case_runner.mode;
      if (cfg.optimization.case_runner.runner) cr.runner = cfg.optimization.case_runner.runner;
      if (cfg.optimization.case_runner.trial_destination) cr.trial_destination = cfg.optimization.case_runner.trial_destination;
      if (cfg.optimization.case_runner.artifacts_folder) cr.artifacts_folder = cfg.optimization.case_runner.artifacts_folder;
      if (cfg.optimization.case_runner.mode === 'remote') {
        if (cfg.optimization.case_runner.remote_status_query) cr.remote_status_query = cfg.optimization.case_runner.remote_status_query;
        if (cfg.optimization.case_runner.remote_early_stop) cr.remote_early_stop = cfg.optimization.case_runner.remote_early_stop;
      }
      if (cfg.optimization.case_runner.variable_substitution?.length) {
        cr.variable_substitution = cfg.optimization.case_runner.variable_substitution;
      }
      if (cfg.optimization.case_runner.file_substitution?.length) {
        cr.file_substitution = cfg.optimization.case_runner.file_substitution;
      }
      if (Object.keys(cr).length > 1) opt.case_runner = cr;
      out.optimization = opt;

      // Orchestration
      const orch = {
        max_trials: cfg.orchestration_settings.max_trials,
        parallelism: cfg.orchestration_settings.parallelism,
      };
      if (cfg.orchestration_settings.tolerated_trial_failure_rate !== 0.5) {
        orch.tolerated_trial_failure_rate = cfg.orchestration_settings.tolerated_trial_failure_rate;
      }
      if (cfg.orchestration_settings.timeout_hours) orch.timeout_hours = cfg.orchestration_settings.timeout_hours;
      if (cfg.orchestration_settings.ttl_seconds_for_trials) orch.ttl_seconds_for_trials = cfg.orchestration_settings.ttl_seconds_for_trials;
      if (cfg.orchestration_settings.initial_seconds_between_polls !== 10) {
        orch.initial_seconds_between_polls = cfg.orchestration_settings.initial_seconds_between_polls;
      }
      if (cfg.orchestration_settings.seconds_between_polls_backoff_factor !== 1.0) {
        orch.seconds_between_polls_backoff_factor = cfg.orchestration_settings.seconds_between_polls_backoff_factor;
      }
      // Global stopping
      const gs = cfg.orchestration_settings.global_stopping_strategy;
      if (gs.min_trials || gs.window_size || gs.improvement_bar) {
        orch.global_stopping_strategy = gs;
      }
      // Early stopping (recursive)
      const buildEs = (node) => {
        if (!node || !node.type) return null;
        const out = { type: node.type };
        if (node.type === 'percentile' || node.type === 'threshold') {
          if (node.metric_names?.length) out.metric_names = node.metric_names;
          if (node.type === 'percentile' && node.percentile_threshold != null) out.percentile_threshold = node.percentile_threshold;
          if (node.type === 'threshold' && node.threshold != null) out.threshold = node.threshold;
          if (node.min_progression) out.min_progression = node.min_progression;
        } else if (node.type === 'and' || node.type === 'or') {
          const left = buildEs(node.left);
          const right = buildEs(node.right);
          if (left) out.left = left;
          if (right) out.right = right;
        }
        return out;
      };
      const esOut = buildEs(cfg.orchestration_settings.early_stopping_strategy);
      if (esOut) orch.early_stopping_strategy = esOut;
      // Dimensionality reduction (omit entirely if disabled)
      const dr = cfg.orchestration_settings.dimensionality_reduction;
      if (dr && dr.enabled) {
        const drOut = { enabled: true };
        if (dr.after_trials !== 10) drOut.after_trials = dr.after_trials;
        if (dr.min_importance !== 0.05) drOut.min_importance = dr.min_importance;
        if (dr.fix_at !== 'best') drOut.fix_at = dr.fix_at;
        if (dr.max_fix_fraction !== 0.5) drOut.max_fix_fraction = dr.max_fix_fraction;
        orch.dimensionality_reduction = drOut;
      }
      out.orchestration_settings = orch;

      // Store
      out.store = {
        save_to: cfg.store.save_to,
        read_from: cfg.store.read_from,
      };
      if ((cfg.store.save_to === 'sql' || cfg.store.read_from === 'sql') && cfg.store.backend_options?.url) {
        out.store.backend_options = { url: cfg.store.backend_options.url };
      }

      // Trial dependencies
      if (cfg.trial_dependencies?.length) {
        out.trial_dependencies = cfg.trial_dependencies.map(dep => {
          const d = { name: dep.name, enabled: dep.enabled };
          const src = { strategy: dep.source.strategy, fallback: dep.source.fallback };
          if (dep.source.strategy === 'by_index' && dep.source.index != null) src.index = dep.source.index;
          if (dep.source.strategy === 'matching_group' && dep.source.group) src.group = dep.source.group;
          if (dep.source.strategy === 'nearest' && dep.source.similarity_threshold != null) src.similarity_threshold = dep.source.similarity_threshold;
          if (dep.source.strategy === 'custom' && dep.source.command) src.command = dep.source.command;
          d.source = src;
          d.actions = dep.actions.map(a => {
            const act = { type: a.type, command: a.command };
            if (a.phase !== 'immediate') act.phase = a.phase;
            return act;
          });
          return d;
        });
      }

      return out;
    },

    // --- Section helpers ---

    setSection(id) {
      this.activeSection = id;
    },

    sectionHasData(id) {
      switch (id) {
        case 'experiment': return !!this.config.experiment.name;
        case 'trial_generation': return this.config.trial_generation.method !== 'fast';
        case 'baseline': return Object.keys(this.config.baseline.parameters).some(k => this.config.baseline.parameters[k] != null && this.config.baseline.parameters[k] !== '');
        case 'optimization': return !!this.config.optimization.objective;
        case 'orchestration': return this.config.orchestration_settings.max_trials !== 50;
        case 'store': return this.config.store.save_to !== 'json' || this.config.store.read_from !== 'nowhere';
        case 'advanced': return this.config.trial_dependencies?.length > 0;
        default: return false;
      }
    },

    // --- Parameter helpers ---

    addParameter() {
      this.config.experiment.parameters.push({
        name: '',
        kind: 'range',
        parameter_type: 'float',
        bounds_lower: 0,
        bounds_upper: 1,
        scaling: '',
        step_size: null,
        values: [],
        groups: [],
      });
    },

    removeParameter(idx) {
      this.config.experiment.parameters.splice(idx, 1);
    },

    addConstraint(e) {
      const val = e.target.value.trim();
      if (val) {
        this.config.experiment.parameter_constraints.push(val);
        e.target.value = '';
      }
    },

    // --- Generation node helpers ---

    addGenNode() {
      this.config.trial_generation.generation_nodes.push({
        node_name: '',
        generator_specs: [{ generator_enum: 'SOBOL', model_kwargs: null }],
        transition_criteria: [],
      });
    },

    addGenSpec(node) {
      node.generator_specs.push({ generator_enum: 'BOTORCH_MODULAR', model_kwargs: null });
    },

    addTransition(node) {
      node.transition_criteria.push({ type: 'max_trials', value: 10 });
    },

    // --- Trial dependency helpers ---

    addTrialDep() {
      this.config.trial_dependencies.push({
        name: '',
        enabled: true,
        source: { strategy: 'best', fallback: 'skip', index: null, group: null, command: null, similarity_threshold: null },
        actions: [{ type: 'run_command', command: '', phase: 'immediate' }],
      });
    },

    addTrialAction(dep) {
      dep.actions.push({ type: 'run_command', command: '', phase: 'immediate' });
    },

    // --- Optimization helpers ---

    addMetric() {
      this.config.optimization.metrics.push({
        name: '',
        command: [],
        progress: [],
        lower_is_better: false,
      });
    },

    addOutcomeConstraint(e) {
      const val = e.target.value.trim();
      if (val) {
        this.config.optimization.outcome_constraints.push(val);
        e.target.value = '';
      }
    },

    addObjThreshold(e) {
      const val = e.target.value.trim();
      if (val) {
        this.config.optimization.objective_thresholds.push(val);
        e.target.value = '';
      }
    },

    addVarSubst() {
      this.config.optimization.case_runner.variable_substitution.push({
        file: '',
        parameter_scopes: {},
      });
    },

    addFileSubst() {
      this.config.optimization.case_runner.file_substitution.push({
        parameter: '',
        file_path: '',
      });
    },

    tryParseJson(str, fallback) {
      try { return JSON.parse(str); } catch { return fallback; }
    },

    // --- Export / Load ---

    exportYaml() {
      const yaml = this.yamlOutput;
      const blob = new Blob([yaml], { type: 'text/yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (this.config.experiment.name || 'config') + '.yaml';
      a.click();
      URL.revokeObjectURL(url);
    },

    loadYaml(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = jsyaml.load(e.target.result);
          this.importConfig(data);
        } catch (err) {
          alert('Failed to parse YAML: ' + err.message);
        }
      };
      reader.readAsText(file);
    },

    importConfig(data) {
      if (!data || typeof data !== 'object') return;

      // Experiment
      if (data.experiment) {
        this.config.experiment.name = data.experiment.name || '';
        this.config.experiment.description = data.experiment.description || '';
        this.config.experiment.parameter_constraints = data.experiment.parameter_constraints || [];
        this.config.experiment.parameters = (data.experiment.parameters || []).map(p => ({
          name: p.name || '',
          kind: p.values ? 'choice' : 'range',
          parameter_type: p.parameter_type || 'float',
          bounds_lower: p.bounds?.[0] ?? 0,
          bounds_upper: p.bounds?.[1] ?? 1,
          scaling: p.scaling || '',
          step_size: p.step_size || null,
          values: p.values || [],
          groups: p.groups || [],
        }));
      }

      // Trial generation
      if (data.trial_generation) {
        this.config.trial_generation.method = data.trial_generation.method || 'fast';
        this.config.trial_generation.initialization_budget = data.trial_generation.initialization_budget ?? null;
        this.config.trial_generation.initialize_with_center = data.trial_generation.initialize_with_center ?? true;
        this.config.trial_generation.generation_nodes = (data.trial_generation.generation_nodes || []).map(n => ({
          node_name: n.node_name || '',
          generator_specs: (n.generator_specs || []).map(s => ({
            generator_enum: s.generator_enum || 'SOBOL',
            model_kwargs: s.model_kwargs || null,
          })),
          transition_criteria: (n.transition_criteria || []).map(tc => {
            const key = Object.keys(tc)[0];
            return { type: key, value: typeof tc[key] === 'number' ? tc[key] : 0 };
          }),
        }));
      }

      // Baseline
      if (data.baseline) {
        this.config.baseline.parameters = data.baseline.parameters || {};
      }

      // Optimization
      if (data.optimization) {
        this.config.optimization.objective = data.optimization.objective || '';
        this.config.optimization.outcome_constraints = data.optimization.outcome_constraints || [];
        this.config.optimization.objective_thresholds = data.optimization.objective_thresholds || [];
        this.config.optimization.metrics = (data.optimization.metrics || []).map(m => ({
          name: m.name || '',
          command: m.command || [],
          progress: m.progress || [],
          lower_is_better: m.lower_is_better || false,
        }));
        if (data.optimization.case_runner) {
          const cr = data.optimization.case_runner;
          Object.assign(this.config.optimization.case_runner, {
            template_case: cr.template_case || '',
            mode: cr.mode || 'local',
            runner: cr.runner || '',
            trial_destination: cr.trial_destination || '',
            artifacts_folder: cr.artifacts_folder || './artifacts',
            remote_status_query: cr.remote_status_query || '',
            remote_early_stop: cr.remote_early_stop || '',
            variable_substitution: cr.variable_substitution || [],
            file_substitution: cr.file_substitution || [],
          });
        }
      }

      // Orchestration
      if (data.orchestration_settings) {
        const os = data.orchestration_settings;
        Object.assign(this.config.orchestration_settings, {
          max_trials: os.max_trials ?? 50,
          parallelism: os.parallelism ?? 1,
          tolerated_trial_failure_rate: os.tolerated_trial_failure_rate ?? 0.5,
          timeout_hours: os.timeout_hours ?? 24,
          ttl_seconds_for_trials: os.ttl_seconds_for_trials ?? 600,
          initial_seconds_between_polls: os.initial_seconds_between_polls ?? 10,
          seconds_between_polls_backoff_factor: os.seconds_between_polls_backoff_factor ?? 1.0,
        });
        if (os.global_stopping_strategy) {
          Object.assign(this.config.orchestration_settings.global_stopping_strategy, os.global_stopping_strategy);
        }
        if (os.early_stopping_strategy) {
          this.config.orchestration_settings.early_stopping_strategy = this._importEsNode(os.early_stopping_strategy);
        }
        if (os.dimensionality_reduction) {
          Object.assign(this.config.orchestration_settings.dimensionality_reduction, os.dimensionality_reduction);
        }
      }

      // Store
      if (data.store) {
        this.config.store.save_to = data.store.save_to || 'json';
        this.config.store.read_from = data.store.read_from || 'nowhere';
        if (data.store.backend_options) {
          this.config.store.backend_options = data.store.backend_options;
        }
      }

      // Trial dependencies
      if (data.trial_dependencies) {
        this.config.trial_dependencies = data.trial_dependencies.map(dep => ({
          name: dep.name || '',
          enabled: dep.enabled ?? true,
          source: {
            strategy: dep.source?.strategy || 'best',
            fallback: dep.source?.fallback || 'skip',
            index: dep.source?.index ?? null,
            group: dep.source?.group ?? null,
            command: dep.source?.command ?? null,
            similarity_threshold: dep.source?.similarity_threshold ?? null,
          },
          actions: (dep.actions || []).map(a => ({
            type: a.type || 'run_command',
            command: a.command || '',
            phase: a.phase || 'immediate',
          })),
        }));
      }

    },

    copyYaml() {
      navigator.clipboard.writeText(this.yamlOutput);
      this.copied = true;
      setTimeout(() => { this.copied = false; }, 2000);
    },

    // --- Validation ---

    validate() {
      const errors = [];
      if (!this.config.experiment.name) errors.push('Experiment name is required');
      if (!this.config.experiment.parameters.length) errors.push('At least one parameter is required');
      this.config.experiment.parameters.forEach((p, i) => {
        if (!p.name) errors.push(`Parameter ${i + 1}: name is required`);
        if (p.parameter_type !== 'str' && p.bounds_lower >= p.bounds_upper) {
          errors.push(`Parameter "${p.name}": lower bound must be < upper bound`);
        }
      });
      if (!this.config.optimization.objective) errors.push('Objective expression is required');
      this.validationErrors = errors;
      return errors.length === 0;
    },

    // --- Preflight / Dry Run ---

    async loadBootstrapPreview() {
      const path = this.config.bootstrap;
      this.bootstrapPreview = null;
      this.bootstrapError = '';
      if (!path) return;
      this.bootstrapLoading = true;
      try {
        const res = await fetch('/api/v1/config/bootstrap-preview?path=' + encodeURIComponent(path));
        const data = await res.json();
        if (!res.ok) {
          this.bootstrapError = data.error || ('HTTP ' + res.status);
          return;
        }
        this.bootstrapPreview = data;
        if (this.config.specialize === null) this.config.specialize = {};
      } catch (err) {
        this.bootstrapError = 'Could not reach server: ' + err.message;
      } finally {
        this.bootstrapLoading = false;
      }
    },

    toggleSpecialize(paramName) {
      if (!this.config.specialize) this.config.specialize = {};
      if (paramName in this.config.specialize) {
        delete this.config.specialize[paramName];
      } else {
        const p = (this.bootstrapPreview?.parameters || []).find(x => x.name === paramName);
        const defaultValue = p && p.bounds ? (p.bounds[0] + p.bounds[1]) / 2 : 0;
        this.config.specialize[paramName] = defaultValue;
      }
      // Trigger Alpine reactivity on dict mutation.
      this.config.specialize = { ...this.config.specialize };
    },

    clearBootstrap() {
      this.config.bootstrap = null;
      this.config.specialize = null;
      this.bootstrapPreview = null;
      this.bootstrapError = '';
    },

    async runPreflight() {
      if (!this.validate()) return;
      this.preflightRunning = true;
      try {
        const res = await fetch('/api/v1/config/preflight', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.buildOutput()),
        });
        this.preflightResults = await res.json();
      } catch (err) {
        this.preflightResults = {
          checks: [{ name: 'Connection', status: 'FAIL', detail: 'Could not reach server: ' + err.message }],
        };
      } finally {
        this.preflightRunning = false;
      }
    },

    async runDryRun() {
      if (!this.validate()) return;
      this.dryRunRunning = true;
      try {
        const res = await fetch('/api/v1/config/dry-run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.buildOutput()),
        });
        const data = await res.json();
        this.preflightResults = data;
      } catch (err) {
        this.preflightResults = {
          checks: [{ name: 'Dry Run', status: 'FAIL', detail: 'Could not reach server: ' + err.message }],
        };
      } finally {
        this.dryRunRunning = false;
      }
    },

    // --- Docs / Concepts / Tooltips ---

    async loadFieldHints() {
      try {
        const res = await fetch('/api/v1/config/docs');
        if (!res.ok) return;
        const docs = await res.json();
        // Separate field docs (plain strings) from concepts/tutorials (objects with category)
        const hints = {};
        const entries = [];
        for (const [key, val] of Object.entries(docs)) {
          if (typeof val === 'string') {
            hints[key] = val;
          } else if (val && val.category) {
            const title = key.replace(/^(concept|tutorial)\./, '').replace(/_/g, ' ');
            entries.push({ key, title, category: val.category.toLowerCase(), content: val.content });
          }
        }
        this.fieldHints = hints;
        this.docsEntries = entries;
        // Apply tooltips to form labels after a tick
        this.$nextTick(() => this.applyTooltips());
      } catch (_) { /* offline — no hints */ }
    },

    async loadDocs() {
      if (this.docsEntries.length > 0) return; // already loaded
      this.docsLoading = true;
      await this.loadFieldHints();
      this.docsLoading = false;
    },

    filteredDocs() {
      if (this.docsFilter === 'all') return this.docsEntries;
      return this.docsEntries.filter(e => e.category === this.docsFilter);
    },

    prismYaml(text) {
      const t = text || '';
      try {
        if (typeof Prism !== 'undefined' && Prism.languages.yaml) {
          return Prism.highlight(t, Prism.languages.yaml, 'yaml');
        }
      } catch (_) { /* fall through */ }
      return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    },

    renderMd(text) {
      const t = text || '';
      try {
        if (typeof marked !== 'undefined' && marked.parse) {
          if (!this._markedReady && marked.use) {
            marked.use({ gfm: true, breaks: false, async: false });
            this._markedReady = true;
          }
          return marked.parse(t);
        }
      } catch (e) {
        console.warn('marked.parse failed, using fallback:', e);
      }
      // Minimal fallback: escape, then convert fenced code blocks + headings + paragraphs.
      const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      let html = '';
      const parts = t.split(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g);
      for (let i = 0; i < parts.length; i++) {
        if (i % 3 === 0) {
          // prose
          html += esc(parts[i])
            .replace(/^(#{1,6})\s+(.+)$/gm, (_, h, t2) => `<h${h.length}>${t2}</h${h.length}>`)
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/``([^`]+)``|`([^`]+)`/g, (_, a, b) => `<code>${a||b}</code>`)
            .replace(/\n{2,}/g, '</p><p>');
          html = html.startsWith('<') ? html : '<p>' + html + '</p>';
        } else if (i % 3 === 2) {
          html += `<pre><code>${esc(parts[i])}</code></pre>`;
        }
      }
      return html;
    },

    hint(path) {
      return this.fieldHints[path] || '';
    },

    applyTooltips() {
      const tip = document.getElementById('foambo-tooltip');
      if (!tip) return;
      // Bind tooltip hover handlers once — cancel hide while cursor is on it
      if (!tip._hoverBound) {
        tip.addEventListener('mouseenter', () => {
          if (this._tipHideTimer) { clearTimeout(this._tipHideTimer); this._tipHideTimer = null; }
          this._tipHovered = true;
        });
        tip.addEventListener('mouseleave', () => {
          this._tipHovered = false;
          tip.classList.remove('visible');
        });
        tip._hoverBound = true;
      }
      document.querySelectorAll('[data-hint]').forEach(el => {
        if (el._hintBound) return;  // don't double-bind
        const path = el.getAttribute('data-hint');
        const text = this.fieldHints[path];
        if (!text) return;
        el.classList.add('has-hint');
        el.removeAttribute('title');
        const html = this.renderMd(this._dedent(text));
        const show = (ev) => {
          if (this._tipHideTimer) { clearTimeout(this._tipHideTimer); this._tipHideTimer = null; }
          tip.innerHTML = html;
          tip.classList.add('visible');
          this._positionTip(tip, ev);
        };
        const move = (ev) => {
          if (!this._tipHovered) this._positionTip(tip, ev);
        };
        const hide = () => {
          // Delay hide so cursor can travel into the tooltip
          if (this._tipHideTimer) clearTimeout(this._tipHideTimer);
          this._tipHideTimer = setTimeout(() => {
            if (!this._tipHovered) tip.classList.remove('visible');
          }, 200);
        };
        el.addEventListener('mouseenter', show);
        el.addEventListener('mousemove', move);
        el.addEventListener('mouseleave', hide);
        el._hintBound = true;
      });
    },

    _positionTip(tip, ev) {
      const pad = 14;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      // Measure
      tip.style.left = '0px';
      tip.style.top = '0px';
      const rect = tip.getBoundingClientRect();
      let x = ev.clientX + pad;
      let y = ev.clientY + pad;
      if (x + rect.width > vw - 8) x = ev.clientX - rect.width - pad;
      if (y + rect.height > vh - 8) y = ev.clientY - rect.height - pad;
      if (x < 8) x = 8;
      if (y < 8) y = 8;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    },

    _dedent(text) {
      // Strip common leading whitespace (pydantic docstrings often indented)
      const lines = text.split('\n');
      const indents = lines.filter(l => l.trim()).map(l => l.match(/^(\s*)/)[1].length);
      const min = indents.length ? Math.min(...indents) : 0;
      return min > 0 ? lines.map(l => l.slice(min)).join('\n') : text;
    },
  };
}
