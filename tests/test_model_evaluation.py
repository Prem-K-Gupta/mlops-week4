from iris_pipeline.pipeline import train_pipeline


def test_accuracy():
    model, X_test, y_test = train_pipeline()
    assert model.score(X_test, y_test) >= 0.90
