plugins {
    kotlin("jvm") version "2.4.20" apply false
    id("org.jetbrains.intellij.platform") version "2.18.1" apply false
}

allprojects {
    group = "dev.faktor"
    version = "0.1.0"
    repositories {
        mavenCentral()
    }
}

subprojects {
    apply(plugin = "kotlin")

    extensions.configure<org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension> {
        jvmToolchain(17)
    }

    tasks.withType<Test>().configureEach {
        failOnNoDiscoveredTests = false
    }

    dependencies {
        if (project.name == "backend") {
            "implementation"(project(":shared"))
            "testImplementation"(kotlin("test"))
        }
    }
}
