export default class TransformersNLP {
  async zeroShotClassification(text: string, labels: string[]): Promise<any> {
    try {
      const { pipeline } = await import("@xenova/transformers");
      const classifier = await pipeline(
        "zero-shot-classification",
        // "Xenova/mobilebert-uncased-mnli",
        "Xenova/nli-deberta-v3-xsmall"
      );
      const output = await classifier(text, labels, { multi_label: true });
      return output;
    } catch (error) {
      throw error;
    }
  }

  async translation(text: string): Promise<any> {
    try {
      const { pipeline } = await import("@xenova/transformers");
      const translator = await pipeline("translation", "Xenova/m2m100_418M");
      const output = await translator(text, {
        // src_lang: "spa_Latn",
        // tgt_lang: "eng_Latn",
        src_lang: "es",
        tgt_lang: "en",
      });
      return output;
    } catch (error) {
      throw error;
    }
  }
}
