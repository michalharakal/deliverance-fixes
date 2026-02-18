package io.teknek.deliverance.vibrant;

import java.nio.file.Path;

interface MessageTransformer {
    void transform(String raw, Path base);
}
