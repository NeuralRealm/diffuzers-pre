from diffusers import DiffusionPipeline

#Return a CheckpointMergerPipeline class that allows you to merge checkpoints. 
#The checkpoint passed here is ignored. But still pass one of the checkpoints you plan to 
#merge for convenience
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", custom_pipeline="checkpoint_merger")

#There are multiple possible scenarios:
#The pipeline with the merged checkpoints is returned in all the scenarios

#Compatible checkpoints a.k.a matched model_index.json files. Ignores the meta attributes in model_index.json during comparision.( attrs with _ as prefix )
merged_pipe = pipe.merge(["CompVis/stable-diffusion-v1-4","CompVis/stable-diffusion-v1-2"], interp = "sigmoid", alpha = 0.4)

prompt = "An astronaut riding a horse on Mars"

image = merged_pipe(prompt).images[0]
image.save('imagic_image_alpha_2.png')