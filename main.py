from src.data_cleaning import clean
from src.models import linear_regression, test_model, random_forest, xgboost_model, compare_models, shap_analysis, vif_analysis, gdp_analysis
from src.config import SEPARATOR_WIDTH

def print_menu():
    print("\n" + "=" * SEPARATOR_WIDTH)
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
    print("11. GDP Dependency Analysis")
    print("0. Exit")

def main():
    while True:
        print_menu()
        choice = input("\nSelect operation (0-11): ").strip()

        if choice == '0':
            print("\nExiting...")
            break
        elif choice == '1':
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
        elif choice == '11':
            gdp_analysis.main()
        else:
            print("Invalid choice. Please select 0-11.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
