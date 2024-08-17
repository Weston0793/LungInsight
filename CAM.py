# Function to get CAM
def get_cam(model, img_tensor, target_layer_name):
    model.eval()
    
    def forward_hook(module, input, output):
        activation[0] = output

    activation = {}
    layer = dict([*model.named_modules()]).get(target_layer_name, None)
    if layer is None:
        raise ValueError(f"Layer {target_layer_name} not found in the model")
        
    hook = layer.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    hook.remove()
    
    weight_softmax_params = list(model.parameters())[-2].detach().numpy()
    weight_softmax = np.squeeze(weight_softmax_params)
    
    activation = activation[0].squeeze().cpu().data.numpy()
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weight_softmax[predicted_class]):
        cam += w * activation[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam
