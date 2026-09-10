import os
import sys
import uuid
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Ensure NST_Code directory is in sys.path so utils can be imported in any context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import existing AdaIN code
try:
    from utils.models import VGGEncoder, Decoder
    from utils.utils import adaptive_instance_normalization
except ImportError:
    from NST_Code.utils.models import VGGEncoder, Decoder  # type: ignore
    from NST_Code.utils.utils import adaptive_instance_normalization  # type: ignore

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['WTF_CSRF_ENABLED'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
# Support all image formats supported by PIL plus common variants
ALLOWED_EXTENSIONS = {
    ext.lstrip('.').lower() for ext in Image.registered_extensions().keys()
}
ALLOWED_EXTENSIONS.update({
    'png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tiff', 'tif', 'jfif',
    'ico', 'svg', 'heic', 'heif', 'avif', 'ppm', 'pgm'
})
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(os.path.join(BASE_DIR, 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth'),
        map_location=device,
        weights_only=True,
    )
)

encoder.eval()
decoder.eval()


def allowed_file(filename):
    if not filename:
        return False
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in app.config['ALLOWED_EXTENSIONS']:
            return True
    return False


def is_valid_image(file_storage):
    """Verify that an uploaded file is a valid image by extension or content inspection."""
    if allowed_file(file_storage.filename):
        return True
    try:
        pos = file_storage.tell()
        file_storage.seek(0)
        with Image.open(file_storage) as img:
            img.verify()
        file_storage.seek(pos)
        return True
    except Exception:
        try:
            file_storage.seek(0)
        except Exception:
            pass
        return False


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm(meta={'csrf': False})
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        content_file = request.files.get('content')
        style_file = request.files.get('style')

        # Handle content image upload or fallback to existing path
        if content_file and content_file.filename:
            if is_valid_image(content_file):
                content_filename = secure_filename(content_file.filename)
                if not content_filename:
                    ext = content_file.filename.rsplit('.', 1)[1].lower() if '.' in content_file.filename else 'png'
                    content_filename = f"content_{uuid.uuid4().hex[:8]}.{ext}"
                content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
            else:
                error = 'Invalid content image file. Please upload an image.'
        elif form.content_path.data:
            content_filename = form.content_path.data

        # Handle style image upload or fallback to existing path
        if style_file and style_file.filename:
            if is_valid_image(style_file):
                style_filename = secure_filename(style_file.filename)
                if not style_filename:
                    ext = style_file.filename.rsplit('.', 1)[1].lower() if '.' in style_file.filename else 'png'
                    style_filename = f"style_{uuid.uuid4().hex[:8]}.{ext}"
                style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
            else:
                error = 'Invalid style image file. Please upload an image.'
        elif form.style_path.data:
            style_filename = form.style_path.data

        if content_filename and style_filename and not error:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            if os.path.exists(content_path) and os.path.exists(style_path):
                try:
                    with Image.open(content_path) as c_img:
                        content_image = c_img.convert('RGB')
                    with Image.open(style_path) as s_img:
                        style_image = s_img.convert('RGB')

                    alpha = float(form.alpha.data) if form.alpha.data is not None else 1.0
                    stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                    # Always save stylized output as PNG for universal browser rendering
                    base_name = os.path.splitext(content_filename)[0]
                    result_filename = f"stylized_{base_name}.png"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    save_image(stylized_image, result_path)

                    result_image = result_filename
                    # Persist filenames so modifying the alpha slider works without re-uploading
                    form.content_path.data = content_filename
                    form.style_path.data = style_filename
                except Exception as e:
                    error = str(e)
            else:
                if not os.path.exists(content_path):
                    error = 'Uploaded content image not found. Please upload again.'
                    content_filename = None
                if not os.path.exists(style_path):
                    error = 'Uploaded style image not found. Please upload again.'
                    style_filename = None
        elif not error:
            if not content_filename and not style_filename:
                error = 'Please upload both content and style images.'
            elif not content_filename:
                error = 'Please upload a content image.'
            elif not style_filename:
                error = 'Please upload a style image.'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
