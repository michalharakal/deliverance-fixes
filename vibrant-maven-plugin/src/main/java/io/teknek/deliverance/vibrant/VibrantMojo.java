package io.teknek.deliverance.vibrant;

import com.google.j2objc.annotations.Property;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugin.MojoFailureException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;
import org.apache.maven.project.MavenProject;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Mojo(name = "generate", defaultPhase = LifecyclePhase.GENERATE_SOURCES)
public class VibrantMojo extends AbstractMojo {

    @Parameter(defaultValue = "${project}", required = true, readonly = true)
    MavenProject project;

    @Parameter(property = "outputDirectory", defaultValue = "${project.build.directory}/generated-sources/vibrant")
    private File outputDirectory;

    @Parameter(property = "overwrite", defaultValue = "false")
    private boolean overwrite;

    //@Parameter(property = "vibe-specs")
    //private List<VibeSpec> vibeSpecs = new ArrayList<>();
    @Parameter(property = "vibeSpecs")
    private VibeSpecs vibeSpecs;

    @Override
    public void execute() throws MojoExecutionException, MojoFailureException {
        System.out.println("Project: " + project.getGroupId() + ":" + project.getArtifactId() + ":" + project.getVersion());
        System.out.println("vibe specs: "+ vibeSpecs);
        System.out.println("overwrite: " + overwrite);
    }
}
