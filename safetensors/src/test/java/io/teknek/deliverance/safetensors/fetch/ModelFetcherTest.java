package io.teknek.deliverance.safetensors.fetch;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;

public class ModelFetcherTest {
    @Test
    void downloadAModel(){
        //given a modelname
        ModelFetcher fetch = new ModelFetcher("tjake", "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4");
        //when i try maybe to downlad the model
        File f = fetch.maybeDownload();
        //then the directory exists
        Assertions.assertTrue(f.exists() && f.isDirectory());
    }
}
