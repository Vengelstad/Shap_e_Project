# Shap-E Project Examples

This directory contains examples and documentation for using the Shap-E 3D object generator.

## Example Outputs

This project demonstrates generating 3D objects using OpenAI's Shap-E model. Below are examples of what you can create:

### Text-to-3D Examples

1. **Example 1: Red Vintage Car**
   - Prompt: "a red vintage car"
   - Output: `output_text_1.ply`
   - Animation: `output_text_1.gif`
   - Description: A 3D model of a classic red car generated from text

2. **Example 2: Wooden Chair**
   - Prompt: "a wooden chair"
   - Output: `output_text_2.ply`
   - Animation: `output_text_2.gif`
   - Description: A simple wooden chair model

3. **Example 3: Coffee Mug with Steam**
   - Prompt: "a coffee mug with steam"
   - Output: `output_text_3.ply`
   - Animation: `output_text_3.gif`
   - Description: A coffee mug with visible steam effect

### Image-to-3D Example

4. **Example 4: From Image**
   - Input: Custom image file
   - Output: `output_from_image.ply`
   - Animation: `output_from_image.gif`
   - Description: 3D reconstruction from a 2D image

### Batch Generation Example

5. **Example 5: Multiple Variations**
   - Prompt: "a colorful mushroom"
   - Outputs: `batch_output_1.ply`, `batch_output_2.ply`, `batch_output_3.ply`, `batch_output_4.ply`
   - Description: Four different variations of the same prompt

## How to Use

### Prerequisites

```bash
# Clone the Shap-E repository
git clone https://github.com/openai/shap-e.git
cd shap-e

# Install the package
pip install -e .

# Install additional dependencies
pip install torch torchvision
pip install pillow
```

### Running the Examples

1. **Text-to-3D Generation**:
   ```python
   from shap_e_implementation import ShapEGenerator
   
   generator = ShapEGenerator(use_gpu=True)
   latents = generator.generate_from_text(
       prompt="a red vintage car",
       output_path="my_car.ply"
   )
   ```

2. **Image-to-3D Generation**:
   ```python
   generator = ShapEGenerator(use_gpu=True)
   latents = generator.generate_from_image(
       image_path="input.jpg",
       output_path="output.ply"
   )
   ```

3. **Rendering as Animation**:
   ```python
   images = generator.render_latents(latents)
   generator.save_as_gif(images[0], "animation.gif")
   ```

## File Formats

- **PLY (Polygon File Format)**: The generated 3D meshes are saved in PLY format, which can be opened in:
  - Blender (free, open-source 3D software)
  - MeshLab (mesh processing software)
  - Online viewers like: https://3dviewer.net/

- **GIF**: Animated previews showing the 3D object rotating

## Parameters Explained

### guidance_scale
- Controls how closely the output matches your prompt
- Higher values (e.g., 15.0) = more faithful to prompt
- Lower values (e.g., 3.0) = more creative variations
- Recommended: 15.0 for text-to-3D, 3.0 for image-to-3D

### batch_size
- Number of variations to generate at once
- Higher values require more GPU memory
- Recommended: 1-4 for most systems

### num_inference_steps
- Number of diffusion steps
- More steps = better quality but slower
- Recommended: 64 for good balance

### rendering_mode
- 'nerf': Neural Radiance Fields (higher quality)
- 'stf': Signed Transform Fields (faster)

## Tips for Best Results

1. **Be Specific**: More detailed prompts produce better results
   - Good: "a red sports car with black wheels"
   - Better: "a shiny red Ferrari sports car with black racing wheels and chrome details"

2. **Use Clear Images**: For image-to-3D, use:
   - Well-lit photos
   - Clear subject with minimal background
   - Front or 3/4 view works best

3. **Experiment**: Try different guidance_scale values to find what works best

4. **GPU Recommended**: While CPU works, GPU is much faster
   - CUDA-enabled NVIDIA GPU recommended
   - Google Colab offers free GPU access

## Common Issues and Solutions

### Issue: Out of Memory
**Solution**: Reduce batch_size to 1, or use CPU instead of GPU

### Issue: Model files not downloading
**Solution**: Check internet connection, models download automatically on first use

### Issue: PLY files won't open
**Solution**: Install Blender (free) or use an online 3D viewer

## Further Resources

- [Original Shap-E Repository](https://github.com/openai/shap-e)
- [Shap-E Paper](https://arxiv.org/abs/2305.02463)
- [Blender Download](https://www.blender.org/download/)
- [Google Colab Notebooks](https://colab.research.google.com/)

## License

This implementation is based on OpenAI's Shap-E project. Please refer to the original repository for licensing information.
