package io.teknek.deliverance.vibrant;

import com.codahale.metrics.MetricRegistry;
import com.google.j2objc.annotations.Property;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugin.MojoFailureException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;
import org.apache.maven.project.MavenProject;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Mojo(name = "generate", defaultPhase = LifecyclePhase.GENERATE_SOURCES)
public class VibrantMojo extends AbstractMojo {

    @Parameter(defaultValue = "${project}", required = true, readonly = true)
    MavenProject project;

    @Parameter(property = "outputDirectory", defaultValue = "${project.build.directory}/generated-sources/vibrant")
    private File outputDirectory;

    @Parameter(property = "overwrite", defaultValue = "false")
    private boolean overwrite;

    @Parameter
    private List<VibeSpec> vibeSpecs;

    @Parameter(name="modelConfig")
    private ModelConfig modelConfig;


    public void runSingleSpec(AbstractModel model, VibeSpec spec) throws MojoExecutionException {
        if (!spec.enabled) {
            return;
        }
        File specDir;
        if (spec.getGenerateTo() == null) {
            spec.setGenerateTo("generated-source");
        }
        if ("generated-source".equalsIgnoreCase(spec.getGenerateTo())) {
            boolean made = outputDirectory.mkdirs();
            specDir = new File(outputDirectory, spec.id);
            boolean made2 = specDir.mkdirs();
            project.addCompileSourceRoot(specDir.getAbsolutePath());
            System.out.println("spec target" + specDir);
        } else if ("existing-source".equalsIgnoreCase(spec.getGenerateTo())) {
            List<String> compileSourceRoots = project.getCompileSourceRoots();
            specDir = new File(compileSourceRoots.get(0));
        } else {
            throw new MojoExecutionException("unsupported mode " + spec.getGenerateTo());
        }

        PromptSupport.Builder b = model.promptSupport().get().builder();
        spec.userMessages.forEach(b::addUserMessage);
        spec.systemMessages.forEach(b::addSystemMessage);
        var uuid = UUID.randomUUID();
        Response k = model.generate(uuid, b.build(), new GeneratorParameters()
                .withNtokens(512)
                .withIncludeStopStrInOutput(false)
                .withStopWords(List.of("<|eot_id|>"))
                .withTemperature(0.2f)
                .withSeed(99998), (s1, f1) -> {
        });
        System.out.println("k: " + k.responseText);
        Path p = Path.of(specDir.toURI());
        Path child = p.resolve("raw.txt");
        JavaResponseTransformer t = new JavaResponseTransformer();
        t.transform(k.responseText, p);
        try {
            Files.writeString(child, k.responseText);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

    }



    @Override
    public void execute() throws MojoExecutionException, MojoFailureException {
        if (modelConfig == null) {
            modelConfig = new ModelConfig();
        }
        boolean anyEnabled = vibeSpecs.stream().anyMatch(VibeSpec::isEnabled);
        if (!anyEnabled){
            System.out.println("no vibespecs are enabled");
            return;
        }

        ModelFetcher fetch = new ModelFetcher(modelConfig.getOwner(), modelConfig.getModelName());
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        DType working = DType.valueOf(modelConfig.getWorkingMemType());
        DType quantized = DType.valueOf(modelConfig.getQuantizedMemType());
        debudConfig();
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel m = ModelSupport.loadModel(f, working, quantized, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
            for (VibeSpec spec : vibeSpecs) {
                runSingleSpec(m, spec);
            }
        }
    }

    private void debudConfig(){
        System.out.println("Project: " + project.getGroupId() + ":"
                + project.getArtifactId() + ":" + project.getVersion());
        System.out.println("vibe specs: " + vibeSpecs);
        System.out.println("overwrite: " + overwrite);
        System.out.println("modelConfig: " + modelConfig);
    }
}
