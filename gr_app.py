## put it all into gradio or smth
import gradio as gr
import torch
import  model_nn, data_setup

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIDDEN_NURONS = 128
CLASS_NAMES, ___, __, _, num_features, MAX_VALS = data_setup.setup_data(1, 0, DEVICE)

print(CLASS_NAMES)

model = model_nn.define_nn_arch(HIDDEN_NURONS, num_features).to(DEVICE)
model.load_state_dict(torch.load('models/binary_rice_classfication_ai_model.pth', map_location=DEVICE))
model.eval()

def predict(area, major, minor, ecc, convx, equiv, ext, peri, rndn, asr):

    model_features = [
        area / MAX_VALS['Area'],
        major / MAX_VALS['MajorAxisLength'],
        minor / MAX_VALS['MinorAxisLength'],
        ecc / MAX_VALS['Eccentricity'],
        convx / MAX_VALS['ConvexArea'],
        equiv / MAX_VALS['EquivDiameter'],
        ext / MAX_VALS['Extent'],
        peri / MAX_VALS['Perimeter'],
        rndn/ MAX_VALS['Roundness'],
        asr / MAX_VALS['AspectRation'],
    ]


    input_tens = torch.tensor([model_features], dtype=torch.float).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tens)
        proba = torch.sigmoid(logits).item()

    return {"Jasmine": 1 -proba, "Gonen": proba}

# for UI
input_list = [
    gr.Number(label="Area"),
    gr.Number(label="Major Axis Length"),
    gr.Number(label="Minor Axis Length"),
    gr.Number(label="Eccentricity"),
    gr.Number(label="Convex  Area"),
    gr.Number(label="Equiv Diameter"),
    gr.Number(label="Extent"),
    gr.Number(label="Perimeter"),
    gr.Number(label="Roundness"),
    gr.Number(label="Aspect Ration"),
]

interface = gr.Interface(
    fn=predict,
    inputs=input_list,
    outputs=gr.Label(num_top_classes=2),
    title="Rice classfication tool",
    description="Put some hypothetical information about ur rice and it can classify what typa rice u got! If u want to try it out but have random values just put 67 multiple times!!!"
)

if __name__ == "__main__":
    interface.launch(share=True)



