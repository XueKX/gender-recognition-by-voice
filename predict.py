from trains import GenderClassifier, device, load_and_preprocess_data, test_loader
import torch
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking
from torchvision import transforms

model = GenderClassifier()
model.load_state_dict(torch.load("model_epoch_19_acc_0.9721.pth"))
classes = {0: "male", 1: "female"}


# Preprocess the new audio data
audio_path = "/datasets/gender_dataset/train/male/5_3478/5_3478_20170720134029.wav"
audio_path = "/datasets/gender_dataset/train/female/14_3751/14_3751_20170816225243.wav"

# Apply the same preprocessing used during training
transform = transforms.Compose(
    [
        MelSpectrogram(16000),
        FrequencyMasking(freq_mask_param=30),
        TimeMasking(time_mask_param=100),
    ]
)

padded_waveform = load_and_preprocess_data(audio_path)
new_audio_features = transform(padded_waveform).unsqueeze(0)  # Add batch dimension

model.eval()

# Forward pass
with torch.no_grad():
    outputs = model(new_audio_features)

# Get the predicted class
predicted_class = torch.argmax(outputs).item()

print(f"Predicted Class:{classes[predicted_class]}")
