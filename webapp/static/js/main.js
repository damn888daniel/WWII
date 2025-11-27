// Обработка формы и отображение результатов

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileDisplay = document.getElementById('fileDisplay');
    const imagePreview = document.getElementById('imagePreview');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');

    // Обработка выбора файла
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Обновление отображения имени файла
            fileDisplay.innerHTML = `
                <span class="upload-icon">✓</span>
                <span class="upload-text">Selected: ${file.name}</span>
            `;

            // Предпросмотр изображения
            const reader = new FileReader();
            reader.onload = function(event) {
                imagePreview.innerHTML = `
                    <img src="${event.target.result}" alt="Preview">
                `;
            };
            reader.readAsDataURL(file);
        }
    });

    // Drag & Drop
    fileDisplay.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--ink-black)';
    });

    fileDisplay.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--border-color)';
    });

    fileDisplay.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--border-color)';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

    // Отправка формы
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an image file.');
            return;
        }

        const caption = document.getElementById('captionInput').value || 'WWII historical photograph';

        // Показать индикатор загрузки
        loadingIndicator.style.display = 'block';
        resultsSection.style.display = 'none';

        // Создать FormData
        const formData = new FormData();
        formData.append('file', file);
        formData.append('caption', caption);

        try {
            // Отправить запрос
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Analysis failed');
            }

            const result = await response.json();

            // Отобразить результаты
            displayResults(result);

        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Error:', error);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });

    function displayResults(result) {
        const isReal = result.prediction === 'real';
        const probReal = (result.prob_real * 100).toFixed(1);
        const probFake = (result.prob_fake * 100).toFixed(1);
        const confidence = result.confidence.toFixed(1);

        // Обновить штамп и вердикт
        const reportStamp = document.getElementById('reportStamp');
        reportStamp.textContent = isReal ? 'AUTHENTIC' : 'SYNTHETIC';
        reportStamp.className = isReal ? 'report-stamp' : 'report-stamp synthetic';

        // Дата и время анализа
        const now = new Date();
        document.getElementById('reportDate').textContent =
            `ANALYZED: ${now.toLocaleDateString('en-US')} ${now.toLocaleTimeString('en-US')}`;

        // Уверенность
        document.getElementById('confidenceValue').textContent = confidence;

        // Вердикт
        const verdictText = document.getElementById('verdictText');
        verdictText.textContent = isReal ? 'AUTHENTIC HISTORICAL PHOTOGRAPH' : 'AI-GENERATED SYNTHETIC IMAGE';
        verdictText.className = isReal ? 'verdict-text real' : 'verdict-text synthetic';

        // Метрики
        document.getElementById('probReal').textContent = probReal + '%';
        document.getElementById('probFake').textContent = probFake + '%';
        document.getElementById('barReal').style.width = probReal + '%';
        document.getElementById('barFake').style.width = probFake + '%';

        // Интерпретация
        const interpretation = getInterpretation(isReal, confidence);
        document.getElementById('interpretationText').textContent = interpretation;

        // Показать результаты
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function getInterpretation(isReal, confidence) {
        if (isReal) {
            if (confidence >= 95) {
                return `CLASSIFICATION: HIGHLY CONFIDENT - This photograph exhibits strong characteristics of authentic
                WWII-era imagery. The visual features, compositional elements, and contextual markers align with
                genuine historical documentation from the period. Recommended for archival acceptance.`;
            } else if (confidence >= 80) {
                return `CLASSIFICATION: CONFIDENT - The analysis indicates this is likely an authentic historical
                photograph from the WWII period. While some minor uncertainties exist, the overall evidence
                supports authenticity. Further manual verification recommended for critical applications.`;
            } else {
                return `CLASSIFICATION: MODERATE CONFIDENCE - The photograph shows characteristics consistent with
                authentic WWII imagery, but with some ambiguous features. Manual expert review strongly
                recommended before archival classification.`;
            }
        } else {
            if (confidence >= 95) {
                return `CLASSIFICATION: HIGHLY CONFIDENT - Strong indicators of AI-generated synthetic content detected.
                The image exhibits telltale patterns consistent with modern diffusion models (e.g., SDXL, Stable Diffusion).
                Characteristics include unnatural lighting, anatomical inconsistencies, or anachronistic elements.
                NOT RECOMMENDED for historical archives.`;
            } else if (confidence >= 80) {
                return `CLASSIFICATION: CONFIDENT - The analysis detects significant markers of synthetic AI-generated
                content. While the image may resemble WWII-era photographs, it likely contains artificially generated
                elements. Exercise extreme caution before including in historical collections.`;
            } else {
                return `CLASSIFICATION: MODERATE CONFIDENCE - The photograph shows some indicators of synthetic generation,
                though with notable uncertainty. Could be manipulated, colorized, or AI-enhanced historical content.
                Requires expert forensic analysis before archival use.`;
            }
        }
    }
});
