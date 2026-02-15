import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RandomNumberTest {

    @Test
    public void sample() throws IOException {
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        ConfigurableTensorProvider withoutNative = new ConfigurableTensorProvider(tensorCache);
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
            String prompt = "Pick a random number between 0 and 100";
            PromptContext ctx = PromptContext.of(prompt);
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, ctx, new GeneratorParameters().withTemperature(0.0f).withSeed(99999),(s1, f1) -> {});
            System.out.println(k);
            assertEquals("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", k.responseText);
        }
    }

    @Test
    public void calc() throws IOException {
        String modelName = "Llama-3.2-3B-Instruct-JQ4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        ConfigurableTensorProvider withoutNative = new ConfigurableTensorProvider(tensorCache);
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
            String prompt = "Generate a java interface named Shape with a method name calculateArea.";
            PromptContext ctx = m.promptSupport().get().builder()
                    .addSystemMessage("You are an assistant that produces concise, production-grade software.")
                    .addSystemMessage("Output java code.")
                    .addSystemMessage("Refrain from editorializing your reply.")
                    .addSystemMessage("Place text that is not java code in comments.")
                    .addUserMessage("Generate a java interface named Shape with a method named area that returns a double.")
                    .addUserMessage("Generate a java class named Circle that extends the Shape interface.")
                    .build();

            var uuid = UUID.randomUUID();
            Response k = m.generate(uuid, ctx, new GeneratorParameters()
                    .withNtokens(512)
                            .withIncludeStopStrInOutput(false)
                    .withStopWords(List.of("<|eot_id|>"))
                    .withTemperature(0.0f).withSeed(99999),(s1, f1) -> { });

            assertEquals("""
                    import java.lang.Math;

public interface Shape {
    double area();
}

public class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
}""".trim(), k.responseText);
        }
    }

}
