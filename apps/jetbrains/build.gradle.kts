plugins {
    kotlin("jvm") version "2.1.0" apply false
}

allprojects {
    group = "dev.faktor"
    version = "0.1.0-SNAPSHOT"
    repositories {
        mavenCentral()
    }
}

subprojects {
    apply(plugin = "kotlin")

    extensions.configure<org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension> {
        jvmToolchain(17)
    }

    dependencies {
        if (project.name == "backend") {
            "testImplementation"(kotlin("test"))
        }
    }
}
