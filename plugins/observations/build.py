from core.prompting.content import PromptContent, PromptImage


def build(context):
    if context["plugins"].get("observations") is not True:
        return ""
    return PromptContent(tuple(
        PromptImage(image, context.get("image_detail"))
        for image in (context.get("agentview_image"), context.get("wrist_image"))
        if image is not None
    ))
