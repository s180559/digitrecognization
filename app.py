import gradio as gr
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("mnist.h5")


# Define a function to predict the digit
def predict_digit(img):
    # Resize the image to 28x28 pixels
    img = img.reshape((1, 28, 28, 1))
    # Normalize the image
    if img.max() > 1:
        img = img / 255.0
    # Predict the class
    res = model.predict([img])[0]
    # create  result dictionary
    return {str(i): float(res[i]) for i in range(10)}


with gr.Blocks(
    css="style.css",
    theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan"),
) as app:
    # create a header and a para
    with gr.Row():
        gr.Markdown(
            """# MNIST Digit Recognizer
            This app recognizes handwritten digits. The app uses a sketchpad to get the input image.
            
            Model used is a two layered Convolution network, followed by a fully connected layer and a softmax layer.
            """,
        )
    with gr.Row():
        gr.Markdown("## Sketchpad")
        gr.Markdown("## Prediction")
    # create a row
    with gr.Row():
        # create a sketchpad
        sketchpad = gr.Sketchpad(
            shape=(28, 28),
            brush_radius=2,
            elem_id="sketchpad",
            label="Draw a digit here",
        )
        blank_sketchpad = gr.Sketchpad(
            invert_colors=True, brush_radius=2, visible=False
        )
        # create a label
        label = gr.Label(
            num_top_classes=3,
            elem_id="label",
            label="Prediction",
        )

    # create a button
    button = gr.Button("Predict", elem_id="btn_pred")
    # bind the button to predict the digit
    button.click(
        predict_digit,
        inputs=sketchpad,
        outputs=label,
    )

    # create a clear button for sketchpad
    clear_button = gr.Button("Clear", elem_id="btn_clr")
    clear_button.click(lambda a: None, inputs=blank_sketchpad, outputs=sketchpad)

app.launch(share=False)
