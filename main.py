from src.data_cleaning import clean
from src.models import linear_regression, test_model, random_forest, xgboost_model, compare_models, shap_analysis, vif_analysis
from src.config import SEPARATOR_WIDTH

def main():
    print("=" * SEPARATOR_WIDTH)
    print("ML Project - World Data Analysis")
    print("=" * SEPARATOR_WIDTH)

    print("\nAvailable operations:")
    print("1. Clean data")
    print("2. Train Linear Regression")
    print("3. Train Lasso (L1)")
    print("4. Train Ridge (L2)")
    print("5. Train Random Forest")
    print("6. Train XGBoost")
    print("7. Test Model")
    print("8. Compare All Models")
    print("9. SHAP Analysis")
    print("10. VIF Analysis")

    choice = input("\nSelect operation (1-10): ").strip()

    if choice == '1':
        clean.main()
    elif choice == '2':
        linear_regression.main(regularization="none")
    elif choice == '3':
        linear_regression.main(regularization="l1", optimize_hyperparams=True)
    elif choice == '4':
        linear_regression.main(regularization="l2", optimize_hyperparams=True)
    elif choice == '5':
        random_forest.main(optimize_hyperparams=True)
    elif choice == '6':
        xgboost_model.main(optimize_hyperparams=True)
    elif choice == '7':
        test_model.main()
    elif choice == '8':
        compare_models.main()
    elif choice == '9':
        shap_analysis.main()
    elif choice == '10':
        vif_analysis.main()
    else:
        print("Invalid choice. Please select 1-10.")

if __name__ == "__main__":
    main()