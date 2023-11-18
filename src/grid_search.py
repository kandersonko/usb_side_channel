from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def main():

    # Grid search
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [None, 5, 10],
        'min_samples_split': [1, 2],
        'min_samples_leaf': [1, 2, 3]
    }



    print("Grid search")

    # Create a Random Forest classifier
    rf = RandomForestClassifier()

    # Create the grid search object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

    grid_search.fit(X_train_encoded, y_train)

    # Print the best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate the best model on the test set
    best_rf = grid_search.best_estimator_
    accuracy = best_rf.score(X_test_encoded, y_test)
    print("Test Accuracy:", accuracy)

    accuracy, report = evaluate_detection(best_rf, X_train_encoded, y_train, X_test_encoded, y_test, target_names)
    print(report)

    output_file_content += "With grid search\n"
    output_file_content += "Best Parameters: " + str(grid_search.best_params_) + "\n"
    output_file_content += "Test Accuracy: " + str(accuracy) + "\n"
    output_file_content += str(report) + "\n"


if __name__ == '__main__':
    main()
