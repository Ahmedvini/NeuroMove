
import tensorflow as tf
from tensorflow import keras
import models

def print_model_summary(model_name, model_func, **kwargs):
    print("\n" + "="*80)
    print(f"Model: {model_name}")
    print("="*80)
    try:
        model = model_func(**kwargs)
        model.summary()
    except Exception as e:
        print(f"Error instantiating {model_name}: {e}")

def main():
    # Common parameters based on defaults seen in models.py
    # n_classes=4, in_chans=22, in_samples=1125 seems typical for BCIIV 2a
    
    # Models that take n_classes, Chans, Samples (or similar)
    # DB_ATCNet(n_classes, in_chans=22, in_samples=1125, ...)
    print_model_summary("DB_ATCNet", models.DB_ATCNet, n_classes=4, in_chans=22, in_samples=1125)

    # ATCNet(n_classes, in_chans=22, in_samples=1125, ...)
    print_model_summary("ATCNet", models.ATCNet, n_classes=4, in_chans=22, in_samples=1125)
    
    # TCNet_Fusion(n_classes, Chans=22, Samples=1125, ...)
    # Note: Keyword args in definition are Chans, Samples
    print_model_summary("TCNet_Fusion", models.TCNet_Fusion, n_classes=4, Chans=22, Samples=1125)

    # EEGTCNet(n_classes, Chans=22, Samples=1125, ...)
    print_model_summary("EEGTCNet", models.EEGTCNet, n_classes=4, Chans=22, Samples=1125)

    # EEGNeX_8_32(n_timesteps, n_features, n_outputs)
    # Note: Definition is (n_timesteps, n_features, n_outputs) -> (Samples, Chans, n_classes)
    print_model_summary("EEGNeX_8_32", models.EEGNeX_8_32, n_timesteps=1125, n_features=22, n_outputs=4)

    # EEGNet(input_layer, ...) - Wait, the definition in models.py line 470 is:
    # EEGNet(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25)
    # But there is also EEGNet_classifier at 457:
    # EEGNet_classifier(n_classes, Chans=22, Samples=1125, ...)
    # It seems EEGNet takes an input_layer tensor, not simple params?
    # Let's check EEGNet_classifier which seems to wrap it or be the full model.
    print_model_summary("EEGNet_classifier", models.EEGNet_classifier, n_classes=4, Chans=22, Samples=1125)

    # DeepConvNet(nb_classes, Chans=64, Samples=256, ...)
    # Defaults in file: Chans=64, Samples=256. We'll use consistent 22/1125 if possible, 
    # but let's stick to the calling signature.
    print_model_summary("DeepConvNet", models.DeepConvNet, nb_classes=4, Chans=22, Samples=1125)

    # ShallowConvNet(nb_classes, Chans=64, Samples=128, ...)
    print_model_summary("ShallowConvNet", models.ShallowConvNet, nb_classes=4, Chans=22, Samples=1125)

if __name__ == "__main__":
    main()
