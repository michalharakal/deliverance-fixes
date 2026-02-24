package io.teknek.deliverance.model.gemma2;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;

public class Gemma2ModelType implements ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma2Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma2Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return GemmaTokenizer.class;
    }
}