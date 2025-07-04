<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Project Functionalities</title>
<style>
  body {
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    background: #f9f9f9;
    color: #222;
    margin: 2rem auto;
    max-width: 700px;
    padding: 0 1rem;
  }
  h2 {
    color: #0366d6;
    border-bottom: 2px solid #0366d6;
    padding-bottom: 0.3rem;
  }
  .section {
    margin-bottom: 2rem;
  }
  .file-name {
    background-color: #e1ecf4;
    color: #0366d6;
    font-weight: 700;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.5rem;
  }
  ul {
    list-style: inside disc;
    margin: 0;
    padding-left: 1rem;
  }
  ul li {
    margin: 0.3rem 0;
    line-height: 1.4;
  }
</style>
</head>
<body>

<section class="section">
  <h2>🔬 Functionalities Main</h2>
  
  <div>
    <span class="file-name">classifier.py</span>
    <ul>
      <li>Classificação com ResNet50</li>
      <li>Cálculo de SSIM entre imagens</li>
      <li>Funções de processamento de imagem</li>
    </ul>
  </div>

  <div>
    <span class="file-name">image_distortion.py</span>
    <ul>
      <li>Aplicação de ruído gaussiano</li>
      <li>Blur e efeitos de distorção</li>
      <li>Transformações geométricas</li>
    </ul>
  </div>

  <div>
    <span class="file-name">graficos.py</span>
    <ul>
      <li>Gráficos de barras limpos (sem eixo X)</li>
      <li>Violin plots para distribuição</li>
      <li>Mapas de densidade com legenda</li>
    </ul>
  </div>

</section>

</body>
</html>
