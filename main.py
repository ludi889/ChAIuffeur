if __name__ == '__main__':
    print('1.Acquire data (screen + input)\n2.Prepare EfficientNet model\n3.Evaluate model')
    choice = input('Please insert number:')
    if choice == '1':
        import acquiring_data

        acquiring_data.get_data()
    elif choice == '2':
    # TODO It is pretty placeholde'rish for now. Model fitting will come after Reinforcment learning preparations
    # import retinanet_configuration

    # model = retinanet_configuration.training_model()
    # model.save('EfficientNet_model')
    elif choice == '3':
        import evaluating_model
        evaluating_model.evaluate_model()
    else:
        pass
