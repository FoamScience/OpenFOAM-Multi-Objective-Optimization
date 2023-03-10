#!/usr/bin/python3
"""Core functionality for performing paramter variation and optimization on OpenFOAM cases"""

import os, subprocess, hashlib, shutil, logging
from omegaconf import OmegaConf
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import plot_config_to_html, render
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.utils.report.render import render_report_elements

from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

log = logging.getLogger(__name__)

def gen_search_space(cfg):
    """
        Generate a search space for Ax from Hydra config object.
        Looks for "parameters" entry in passed-in config object.
    """

    l = []
    for key, item in cfg.parameters.items():
        e = {
            "name": key,
            **item
        }
        if 'values' in e.keys():
            e['values'] = list(e['values'])
        if 'bounds' in e.keys():
            e['bounds'] = list(e['bounds'])
        if 'dependents' in e.keys():
            tmp = {}
            for k in e['dependents']:
                tmp.update(k)
            e['dependents'] = tmp
        l.append(e)    
    return l

def gen_objectives(cfg):
    """
        Generate objectives for Multi-objective optimization studies from Hydra config object
        Looks for "objectives" entry in passed-in config object.
    """

    objs = {}
    for key, item in cfg.objectives.items():
        objs[key] = ObjectiveProperties(minimize=item.minimize, threshold=item.threshold)
    return objs

def evaluate(parameters, cfg, data):
    """
        Evaluates a single trial (basically an OpenFOAM Case) and populates data dictionary
        with metadata from the trial run
    """

    # Hash parameters to avoid long trial names
    hash = hashlib.md5()
    encoded = repr(OmegaConf.to_yaml(parameters)).encode()
    hash.update(encoded)

    # Clone template case
    templateCase = SolutionDirectory(f"{cfg.problem.template_case}", archive=None, paraviewLink=False)
    for d in cfg.meta.case_subdirs_to_clone:
        templateCase.addToClone(d)
    newcase = f"{cfg.problem.name}_trial_"+hash.hexdigest()
    data["casename"] = newcase

    # Run the case through the provided command in the config
    newcaseExists = os.path.exists(os.path.curdir+"/"+newcase)
    evals = {}
    if not newcaseExists:
        case = templateCase.cloneCase(newcase)
        # Process parameters which require file copying (you can have one parameter per case file)
        if "file_copies" in cfg.meta.keys():
            for elm,elmv in cfg.meta.file_copies.items():
                shutil.copyfile(
                    case.name+elmv.template+"."+parameters[elm],
                    case.name+elmv.template
                )
        # Process parameters with PyFoam
        for elm,elmv in cfg.meta.scopes.items():
            paramFile = ParsedParameterFile(case.name + elm)
            for param in elmv:
                splits = elmv[param].split('.')
                lvl = paramFile[splits[0]]
                if len(splits) > 1:
                    for i in range(1,len(splits)-1):
                        scp = splits[i]
                        try:
                            scp = int(splits[i])
                        except:
                            pass
                        lvl = lvl[scp]
                    lvl[splits[-1]] = parameters[param]
                else:
                    paramFile[elmv[param]] = parameters[param]
            paramFile.writeFile()
        log.info(f"Running trial: {newcase} with paramters:\n{OmegaConf.to_yaml(parameters)}")
        try:
            subprocess.check_output(cfg.meta.case_run_command, cwd=case.name, stderr=subprocess.PIPE, timeout=cfg.meta.case_command_timeout)
        except:
            log.info(f"Case run command {cfg.meta.case_run_command} exited. Setting metrics to None for this trial...")
        for key, item in cfg.problem.objectives.items():
            try:
                evals[key] = float(subprocess.check_output(list(item.command), cwd=case.name))
            except:
                evals[key] = None
    else:
        log.info(f"Case {newcase} exists already, with parameters:\n{OmegaConf.to_yaml(parameters)}\nTrying to grab its metrics...")
        for key, item in cfg.problem.objectives.items():
            try:
                evals[key] = float(subprocess.check_output(list(item.command), cwd=os.path.curdir+"/"+newcase))
            except:
                evals[key] = None
    return evals

def plot_frontier(frontier,CI_level,name):
    """
        Plot pareto frontier with CI_level error bars into an HTML file.
    """

    plot_config = plot_pareto_frontier(frontier, CI_level=0.90)
    with open(f'{name}_report.html', 'w') as outfile:
        outfile.write(render_report_elements(
            f"{name}_report", 
            html_elements=[plot_config_to_html(plot_config)], 
            header=False,
        ))
    render(plot_config)

