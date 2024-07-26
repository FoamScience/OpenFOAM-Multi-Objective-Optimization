#!/usr/bin/env python3

import hydra
from foambo.core import *
from ax.service.ax_client import AxClient, MultiObjective
from ax.core import OptimizationConfig, Experiment, Objective, MultiObjectiveOptimizationConfig
from ax.core.optimization_config import ObjectiveThreshold
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.generation_node import GenerationStep
from ax.service.utils.report_utils import exp_to_df
from ax.storage.json_store.load import load_experiment
from ax.plot.contour import interact_contour
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import plot_objective_vs_constraints

def validation_data_from_experiment(scheduler: Scheduler):
    """
        Produce a CSV file for the trials from the validation run
    """
    cfg = scheduler.experiment.runner.cfg
    params_df = pd.DataFrame()
    exp_df = scheduler.experiment.fetch_data().df
    cl = AxClient().load_from_json_file(f'{cfg.problem.original_name}_client_state.json')
    cl._experiment = scheduler.experiment
    if "trial_index" in exp_df.columns:
        df = exp_to_df(exp=scheduler.experiment)
        trials = scheduler.experiment.get_trials_by_indices(range(df.shape[0]))
        for tr in trials:
            preds = cl.get_model_predictions_for_parameterizations([tr.arm.parameters])
            dt_pred = {}
            for j, (v,s) in preds[0].items():
                dt_pred.update({f"Pred_{j}":v, f"Pred_{j}_sems": s})
            params_df = pd.concat([params_df,
                pd.DataFrame({
                    **dt_pred,
                    **tr._properties},
                    index=[tr.index])])
        df = pd.merge(df, params_df, left_index=True, right_index=True)
        df.set_index("trial_index", inplace=True)
        df.to_csv(f"{cfg['problem']['name']}_validate_report.csv")

@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config.yaml")
def val_main(cfg : DictConfig) -> None:
    if "validate" not in cfg.keys():
        log.info("No validate section found in configuration file. Exiting...")
        return
    cl = AxClient().load_from_json_file(f'{cfg.problem.name}_client_state.json')
    orig_exp = load_experiment(f"{cfg.problem.name}_experiment.json")
    # Mutate the config, but never write it back
    OmegaConf.set_struct(cfg.problem, False)
    cfg.problem.original_name = cfg.problem.name
    cfg.problem.name = "Validate_"+cfg.problem.name
    search_space = gen_search_space(cfg.problem)
    metrics = [HPCJobMetric(name=key, cfg=cfg) for key, _ in cfg.problem.objectives.items()]
    objectives=[Objective(metric=m, minimize=item.minimize) for m, (_, item) in zip(metrics, cfg.problem.objectives.items())]
    thresholds=[ObjectiveThreshold(metric=m, bound=float(item.threshold), relative=False) for m, (_, item) in zip(metrics, cfg.problem.objectives.items())]
    optimization_config = MultiObjectiveOptimizationConfig(objective=MultiObjective(objectives), objective_thresholds=thresholds) \
            if len(objectives) > 1 else OptimizationConfig(objectives[0])
    # Do Ax's cross validation and output few related plots
    if "cross_validate" in cfg.validate.keys() and cfg.validate.cross_validate:
        cl._experiment = orig_exp
        model_spec = cl.generation_strategy._curr.model_spec_to_gen_from
        model_spec.fit(
            orig_exp,
            orig_exp.fetch_data(metrics),
        )
        cur_model = cl.generation_strategy._curr._fitted_model
        for m in metrics:
            ctr = interact_contour(model=cur_model, metric_name=m.name)
            render(ctr)
            with open(f'{cfg.problem.name}_{m.name}_contour.html', 'w') as outfile:
                outfile.write(render_report_elements(
                    f"{cfg.problem.name}_{m.name}_contour", 
                    html_elements=[plot_config_to_html(ctr)], 
                    header=False,
                ))
            obj_vs_constraints = plot_objective_vs_constraints(cur_model, m.name, rel=False)
            render(obj_vs_constraints)
            with open(f'{cfg.problem.name}_{m.name}_objective_vs_constraints.html', 'w') as outfile:
                outfile.write(render_report_elements(
                    f"{cfg.problem.name}_{m.name}_objective_vs_constraints", 
                    html_elements=[plot_config_to_html(obj_vs_constraints)], 
                    header=False,
                ))
        cv_results = cross_validate(cur_model)
        log.info("Cross-validation diagnsotics")
        log.info("============================")
        log.info(pd.DataFrame(compute_diagnostics(cv_results)))
        cv = interact_cross_validation(cv_results)
        render(cv)
        with open(f'{cfg.problem.name}_cross_validation.html', 'w') as outfile:
            outfile.write(render_report_elements(
                f"{cfg.problem.name}_cross_validation", 
                html_elements=[plot_config_to_html(cv)], 
                header=False,
            ))
    if "trials" not in cfg.validate.keys() and "pareto_frontier" not in cfg.validate.keys():
        return
    # Now to the user-supplied trials
    exp = Experiment(
        name=f"{cfg.problem.name}_experiment",
        search_space=cl.make_search_space(parameters=search_space, parameter_constraints=[]),
        optimization_config=optimization_config,
        runner=HPCJobRunner(cfg=cfg),
        is_test=False,
    )
    validate_trials = []
    if "trials" in cfg.validate.keys():
        validate_trials += cfg.validate.trials
    if "pareto_frontier" in cfg.validate.keys() and len(metrics) > 1:
        from ax.plot.pareto_utils import compute_posterior_pareto_frontier
        m1_idx = next((i for i, item in enumerate(metrics) if item.name == cfg.validate.primary_objective), -1)
        m2_idx = next((i for i, item in enumerate(metrics) if item.name == cfg.validate.secondary_objective), -1)
        frontier = compute_posterior_pareto_frontier(
            experiment=orig_exp,
            data=orig_exp.fetch_data(),
            primary_objective=metrics[m1_idx],
            secondary_objective=metrics[m2_idx],
            absolute_metrics=[metrics[m1_idx].name, metrics[m2_idx].name],
            num_points=cfg.validate.pareto_frontier,
        )
        validate_trials = validate_trials + frontier.param_dicts

    gs = GenerationStrategy([
        GenerationStep(
            model=CustomModels.Manual,
            num_trials=len(validate_trials),
            max_parallelism=cfg.meta.n_parallel_trials,
            model_kwargs={
                "parameter_sets": validate_trials,
                "search": cl.make_search_space(parameters=search_space, parameter_constraints=[]),
            },
        )
    ])
    scheduler = Scheduler(
        experiment=exp,
        generation_strategy=gs,
        options=SchedulerOptions(
            log_filepath=log.manager.root.handlers[1].baseFilename,
            ttl_seconds_for_trials=cfg.meta.trial_ttl
                if "trial_ttl" in cfg.meta.keys() else None,
            init_seconds_between_polls=cfg.meta.init_poll_wait
                if "init_poll_wait" in cfg.meta.keys() else 1,
            seconds_between_polls_backoff_factor=cfg.meta.poll_factor
                if "poll_factor" in cfg.meta.keys() else 1.5,
            timeout_hours=cfg.meta.timeout_hours
                if "timeout_hours" in cfg.meta.keys() else None,
        ),
    )
    scheduler.run_n_trials(max_trials=len(validate_trials),
        ignore_global_stopping_strategy=True,
        idle_callback=validation_data_from_experiment)
    scheduler.summarize_final_result()


if __name__ == "__main__":
    val_main()
