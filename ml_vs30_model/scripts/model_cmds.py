from pathlib import Path
import typer

import ml_tools as mlt
import ml_vs30_model as vs30

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False)



@app.command("cv-train-ngboost")
def cv_train_ngboost(config_ffp: Path):
    mlt.utils.setup_logging()

    config = vs30.ModelConfig.from_yaml(config_ffp)

    vs30.ngboost_model.cv_train(config)


@app.command("placeholder")
def placeholder():
    pass

if __name__ == "__main__":
    app()




