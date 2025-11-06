import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main(n_estimators, max_depth, class_weight):
    df = pd.read_csv('../data/winequality-white.csv', sep=';')
    df['quality_bin'] = (df['quality'] >= 7).astype(int)
    X = df.drop(['quality', 'quality_bin'], axis=1)
    y = df['quality_bin']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run():
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", class_weight)
        
        # Log metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_high_quality", report["1"]["f1-score"])
        mlflow.log_metric("f1_low_quality", report["0"]["f1-score"])
        
        # Log artifact
        with open("classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact("classification_report.txt")
        
        # Log model
        mlflow.sklearn.log_model(clf, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--class_weight", type=str, default="balanced")
    args = parser.parse_args()
    
    main(args.n_estimators, args.max_depth, args.class_weight)
