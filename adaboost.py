import numpy as np


# Decision stump used as weak classifier in Adaboost
class DecisionStump:
    def __init__(self):
        self.feature_index = None  # Index of the feature to split on
        self.threshold = None  # Threshold value to split at
        self.polarity = 1  # Polarity of the stump
        self.alpha = None  # Classifier weight

    def predict(self, X):
        # Make predictions for input samples X
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


# Adaboost ensemble classifier
class Adaboost:

    def __init__(self, n_clf=5):
        self.n_clf = n_clf  # Number of weak classifiers

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize sample weights uniformly
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        # Train each weak classifier
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            # Find the best decision stump
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Calculate weighted error
                    missclassified = w[y != predictions]
                    error = sum(missclassified)

                    # Flip polarity if error > 0.5
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Store the best stump
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # Compute classifier weight (alpha)
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Update sample weights
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # Save the trained stump
            self.clfs.append(clf)

    def predict(self, X):
        # Aggregate predictions from all weak classifiers
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
