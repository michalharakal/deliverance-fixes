package io.teknek.deliverance.model;

public class NoOpTokenizerRenderer implements TokenRenderer {
    public String tokenizerToRendered(String token) {
        return token;
    }
}
