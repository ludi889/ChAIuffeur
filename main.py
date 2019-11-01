if __name__ == '__main__':
    print('1.Acquire data (screen + input)\n2.Fit model\n3.Evaluate model')
    choice = input('Please insert number:')
    if choice == '1':
        import acquiring_data

        acquiring_data.get_data()
    elif choice == '2':
        import fitting_model

        fitting_model.fit_model()
    elif choice == '3':
        import evaluating_model

        evaluating_model.evaluate_model()
    else:
        pass
