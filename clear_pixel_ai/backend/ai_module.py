from clear_pixel_ai.backend.ai_models import ai_denoise as model_denoise, ai_restore as model_restore, ai_super_resolve as model_super_resolve, ai_inpaint as model_inpaint, ai_edge_detection as model_edge_detection

def ai_denoise(image):
    return model_denoise(image)

def ai_restore(image):
    return model_restore(image)

def ai_super_resolve(image):
    return model_super_resolve(image)

def ai_inpaint(image, mask=None):
    return model_inpaint(image, mask)

def ai_edge_detection(image):
    return model_edge_detection(image)
