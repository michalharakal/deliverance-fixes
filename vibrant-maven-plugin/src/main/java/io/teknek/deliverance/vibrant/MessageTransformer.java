package io.teknek.deliverance.vibrant;

import java.nio.file.Path;

public interface MessageTransformer {
    void transform(String raw, Path base, VibeSpec vibeSpec);
}
