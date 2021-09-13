if __name__ == '__main__':
    print('1.Acquire data (screen + input)\n2.Evaluate model')
    choice = input('Please insert number:')
    if choice == '1':
        import acquiring_data
        acquiring_data.get_data()
    elif choice == '2':
        import evaluating_model
        evaluating_model.evaluate_model()
    else:
        pass
