import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K

# Load and preprocess images
def load_and_process_img(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_img(img):
    img = img.reshape((224, 224, 3))
    img = img[:, :, ::-1]  # Convert from BGR to RGB
    img += [103.939, 116.779, 123.68]  # Add the mean pixel values
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Load content and style images
content_path = 'path_to_your_content_image.jpg'  # Replace with your content image path
style_path = 'path_to_your_style_image.jpg'      # Replace with your style image path

content_img = load_and_process_img(content_path)
style_img = load_and_process_img(style_path)

# Load VGG19 model
model = vgg19.VGG19(weights='imagenet', include_top=False)

# Define layers for content and style
content_layer = 'block5_conv2'  # Content layer
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

# Create a model that outputs the content and style features
outputs = [model.get_layer(name).output for name in style_layers]
outputs.append(model.get_layer(content_layer).output)
model = tf.keras.Model([model.input], outputs)

# Define loss functions
def get_content_loss(base_content, target):
    return K.sum(K.square(base_content - target))

def get_style_loss(base_style, target):
    S = gram_matrix(base_style)
    C = gram_matrix(target)
    return K.sum(K.square(S - C))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# Compute total loss
def total_loss(outputs, content_weight=1e3, style_weight=1e-2):
    style_outputs = outputs[:-1]
    content_output = outputs[-1]
    
    content_loss = get_content_loss(content_output[0], content_img)
    
    style_loss = K.variable(0.0)
    for style_output in style_outputs:
        style_loss += get_style_loss(style_output[0], style_img)
    
    total_style_loss = style_weight * style_loss
    total_content_loss = content_weight * content_loss
    
    return total_style_loss + total_content_loss

# Create a generated image
generated_img = tf.Variable(content_img, dtype=tf.float32)

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.02)

# Run the optimization
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        outputs = model(generated_img)
        loss = total_loss(outputs)
    grads = tape.gradient(loss, generated_img)
    optimizer.apply_gradients([(grads, generated_img)])
    return loss

# Run the style transfer
num_iterations = 1000
for i in range(num_iterations):
    loss = train_step()
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.numpy()}")

# Display the generated image
final_img = deprocess_img(generated_img.numpy())
plt.imshow(final_img)
plt.axis('off')
plt.show()