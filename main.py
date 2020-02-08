if __name__ == '__main__':
    print('1.Acquire data (screen + input)\n2.Prepare EfficientNet model\n3.Evaluate model')
    choice = input('Please insert number:')
    if choice == '1':
        import acquiring_data

        acquiring_data.get_data()
    elif choice == '2':
        # TODO It is pretty placeholde'rish for now. In latter version should be adjusted for fine-tuning model, but
        # TODO I cannot get along with model shapes. It just works for now.
        # TODO loading saved model currently doesn't work due to swish activation problems. So this option is for now not necessary, because model is compiled just in time
        import retinanet_configuration

        model = retinanet_configuration.trainingmodel()
        model.save('EfficientNet_model')
    elif choice == '3':
        import evaluating_model
        evaluating_model.evaluate_model()
    else:
        pass
