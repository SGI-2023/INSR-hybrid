import os
import imageio
import numpy as np


def create_gif(path):
    # get file names sorted by frame_id
    filenames = sorted((fn for fn in os.listdir(path) if fn.endswith('.png')), key=lambda fn: int(fn[1:4]))

    # create a list to store images
    images = []
    for filename in filenames:
        images.append(imageio.imread(os.path.join(path, filename)))

    return images

def generate_animation(path_result, path_gt, path_error, output_file="advection2D.gif"):

    result_images = create_gif(path_result)
    gt_images = create_gif(path_gt)
    error_images = create_gif(path_error)

    imageio.mimsave("result.gif", result_images)
    imageio.mimsave("gt.gif", gt_images)
    imageio.mimsave("error.gif", error_images)

    num_frames = min(len(result_images), len(gt_images), len(error_images))

    images = []
    for frame in range(num_frames):
        combined_image = np.concatenate([result_images[frame], gt_images[frame], error_images[frame]], axis=1)
        images.append(combined_image)

    imageio.mimsave(output_file, images)


path_result = "values_gif/"

images = create_gif(path_result)
imageio.mimsave("values_constant_w.gif", images)


#generate_animation(path_result, path_gt, path_error, output_file="advection2D.gif")