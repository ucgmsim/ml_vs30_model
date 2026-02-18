from pathlib import Path
import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False)



@app.command("cv-train-ngboost")
def cv_train_ngboost(config_ffp: Path):
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_yaml(config_ffp)

    vs30.ngboost_model.cv_train(run_config)

@app.command("cv-train-catboost")
def cv_train_catboost(run_config_ffp: Path, rel_dataset_ffp: Path, id_suffix: str | None = None):
    mlt.utils.setup_logging()

    run_config = vs30.RunConfig.from_config_kwargs(run_config_ffp, rel_dataset_ffp=rel_dataset_ffp)


    id_suffix = f"_{id_suffix}" if id_suffix is not None else ""
    base_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(True)}{id_suffix}"
    )

    vs30.catboost_model.cv_train(run_config, base_out_dir)

@app.command("placeholder")
def placeholder():
    pass

if __name__ == "__main__":
    app()




