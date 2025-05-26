from nemo.collections.asr.models import EncDecCTCModel

# Replace with the actual path to your .nemo file
model_path = "path/to/your_nemo_file"
model = EncDecCTCModel.restore_from(model_path)

# Export to ONNX format
model.export("app/model/asr_model.onnx")

vocabs = model.decoder.vocabulary

with open("app/model/vocab.txt", "w", encoding="utf-8") as f:
    for token in vocabs:
        f.write(token + "\n")


