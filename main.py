if __name__ == '__main__':
    print('1.Acquire data (screen + input)\n2. Fit model\n3.Evaluate model')
    choice = input('Please choose what you want to do:')
    if choice == 1:
        from acquiring_data import get_data

        get_data()
    elif choice == 2:
        from fitting_model import fit_model

        fit_model()
    elif choice == 3:
        from evaluating_model import evaluate_model

        evaluate_model()
