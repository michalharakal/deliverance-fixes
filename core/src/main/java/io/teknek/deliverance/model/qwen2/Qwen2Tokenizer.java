package io.teknek.deliverance.model.qwen2;

import io.teknek.deliverance.model.llama.LlamaTokenizer;

import java.nio.file.Path;

public class Qwen2Tokenizer extends LlamaTokenizer {
    public Qwen2Tokenizer(Path modelRoot) {
        super(modelRoot);
    }

    @Override
    public String tokenForResponse(String decoded) {
        return decoded;
    }

    /*
    @Override
    protected String postProcessToken(String decoded) {
        String s = super.postProcessToken(decoded);
        if (s.startsWith("Ġ")){
            return " " + s.substring(1);
        }
        return decoded;
    }*/
}
