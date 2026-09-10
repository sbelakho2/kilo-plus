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
        compilerOptions {
            jvmDefault.set(org.jetbrains.kotlin.gradle.dsl.JvmDefaultMode.NO_COMPATIBILITY)
        }
    }

    tasks.withType<Test>().configureEach {
        failOnNoDiscoveredTests = false
    }

    dependencies {
        "compileOnly"("org.jetbrains.kotlin:kotlin-stdlib:1.9.22")

        if (project.name == "backend") {
            "implementation"(project(":shared"))
            "testImplementation"(kotlin("test"))
        }
    }
}
