const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const preview = document.getElementById('preview');
const previewCtx = preview.getContext('2d');
let isDrawing = false;
let model = null;

ctx.strokeStyle = '#000000';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

async function loadModel() {
    try {
        model = await ort.InferenceSession.create('./mnist_model.onnx');
        console.log('Modèle chargé avec succès');
    } catch (error) {
        console.error('Erreur lors du chargement du modèle:', error);
        showError('Impossible de charger le modèle ONNX');
    }
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', drawing);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseout', stopDraw);

canvas.addEventListener('touchstart', handleTouch);
canvas.addEventListener('touchmove', handleTouch);
canvas.addEventListener('touchend', stopDraw);

function startDraw(e) {
    isDrawing = true;
    const pos = getMousePos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function drawing(e) {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    updatePreview();
}

function stopDraw() {
    isDrawing = false;
}

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                    e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function preprocessImage() {
    const small = document.createElement('canvas');
    small.width = 28;
    small.height = 28;
    const smallCtx = small.getContext('2d');
    
    smallCtx.fillStyle = 'black';
    smallCtx.fillRect(0, 0, 28, 28);
    
    smallCtx.globalCompositeOperation = 'screen';
    smallCtx.filter = 'invert(1)';
    smallCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = smallCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        input[i] = pixels[i * 4] / 255.0;
    }
    
    return input;
}

function updatePreview() {
    const data = preprocessImage();
    
    const imageData = previewCtx.createImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
        const val = Math.round(data[i] * 255);
        imageData.data[i * 4] = val;
        imageData.data[i * 4 + 1] = val;
        imageData.data[i * 4 + 2] = val;
        imageData.data[i * 4 + 3] = 255;
    }
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);
    
    previewCtx.clearRect(0, 0, 140, 140);
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(tempCanvas, 0, 0, 140, 140);
}

function clearCanvas() {
    ctx.clearRect(0, 0, 280, 280);
    previewCtx.fillStyle = '#1a202c';
    previewCtx.fillRect(0, 0, 140, 140);
    document.getElementById('result').style.display = 'none';
}

async function predict() {
    if (!model) {
        showError('Le modèle n\'est pas encore chargé');
        return;
    }

    try {
        showLoading();
        
        const inputData = preprocessImage();
        
        const sum = inputData.reduce((a, b) => a + b, 0);
        if (sum < 0.1) {
            showError('Veuillez dessiner un chiffre avant de prédire');
            return;
        }
        
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
        const results = await model.run({ input: tensor });
        const outputs = results.output.data;
        
        let maxIdx = 0;
        for (let i = 1; i < outputs.length; i++) {
            if (outputs[i] > outputs[maxIdx]) {
                maxIdx = i;
            }
        }
        
        const max = Math.max(...outputs);
        const exp = Array.from(outputs).map(x => Math.exp(x - max));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        const probs = exp.map(x => x / sumExp);
        
        const confidence = (probs[maxIdx] * 100).toFixed(1);
        
        showPrediction(maxIdx, confidence, outputs, probs);
        
    } catch (error) {
        console.error('Erreur lors de la prédiction:', error);
        showError('Une erreur est survenue pendant la prédiction');
    }
}

function showPrediction(digit, confidence, scores, probabilities) {
    document.getElementById('prediction').textContent = digit;
    document.getElementById('confidence').textContent = `Confiance: ${confidence}%`;
    
    const debugInfo = `Scores de sortie:
${Array.from(scores).map((x, i) => `${i}: ${x.toFixed(3)}`).join('  ')}

Probabilités:
${probabilities.map((x, i) => `${i}: ${(x*100).toFixed(1)}%`).join('  ')}

Top 3 prédictions:
${probabilities.map((p, i) => ({prob: p, class: i}))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3)
    .map(x => `${x.class} (${(x.prob*100).toFixed(1)}%)`)
    .join(', ')}`;
    
    document.getElementById('debug').textContent = debugInfo;
    document.getElementById('result').style.display = 'block';
}

function showLoading() {
    document.getElementById('prediction').textContent = '...';
    document.getElementById('confidence').innerHTML = '<span class="loading">Analyse en cours</span>';
    document.getElementById('debug').textContent = '';
    document.getElementById('result').style.display = 'block';
}

function showError(message) {
    document.getElementById('prediction').textContent = '!';
    document.getElementById('confidence').innerHTML = `<div class="error">${message}</div>`;
    document.getElementById('debug').textContent = '';
    document.getElementById('result').style.display = 'block';
}

clearCanvas();
loadModel();