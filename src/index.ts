import express from "express";
import TransformersNLP from "./transformersNLP.js";

const app = express();
const port = 3001;
app.use(express.json());

app.post("/classifier", async (req: any, res: any) => {
  try {
    // const { text, labels } = req.body;
    // if (!text || !labels || !Array.isArray(labels)) {
    //   return res.status(400).json({ error: "Se requiere texto y etiquetas" });
    // }
    const text = req.body.text;
    const labels  = req.body.labels;
    const output = await new TransformersNLP().zeroShotClassification(
      text,
      labels
    );
    res.json(output);
  } catch (error) {
    res.status(500).send("Error generando respuesta");
  }
});

app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
});
