package io.teknek.deliverance.model.qwen2;

import io.teknek.deliverance.model.TokenRenderer;

public class Qwen2TokenizerRenderer implements TokenRenderer {
    @Override
    public String tokenizerToRendered(String token) {
      return token.replace('Ġ', ' ')
                    .replace('č', '\n')
              .replace('Ċ', '\n');

    }
}
