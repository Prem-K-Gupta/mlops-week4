from sklearn.datasets import load_iris


def test_schema_and_nulls():
    iris = load_iris(as_frame=True).frame
    expected = [
        "sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)", "target"
    ]
    assert list(iris.columns) == expected
    assert iris.isna().sum().sum() == 0
