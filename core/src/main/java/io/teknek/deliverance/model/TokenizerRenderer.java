package io.teknek.deliverance.model;

//Examples: "Hello how are you" might be tokenized as ["Hello", "Ġhow", "Ġare", "Ġyou"].
class TokenizerRenderer implements TokenRenderer {
    public String tokenizerToRendered(String token) {
        return token.replace('Ġ', ' ')
                .replace('Ċ', '\n');
    }
}
