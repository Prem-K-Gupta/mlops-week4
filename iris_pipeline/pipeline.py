from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_pipeline(test_size: float = 0.2, random_state: int = 42):
    """
    Trains a RandomForest on the IRIS dataset and
    returns (model, X_test, y_test) so unit-tests can validate accuracy.
    """
    iris = load_iris(as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data, iris.target,
        test_size=test_size, random_state=random_state
    )
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te
