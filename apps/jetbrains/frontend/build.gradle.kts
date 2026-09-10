plugins {
    id("org.jetbrains.intellij.platform")
}

intellijPlatform {
    projectName = "faktor"
    buildSearchableOptions = false
    sandboxContainer = rootProject.layout.buildDirectory.dir("intellijPlatform/sandbox")

    pluginConfiguration {
        id = "dev.faktor.jetbrains"
        name = "Faktor"
        version = "0.1.0"
        description = """
            Faktor split-mode bridge for JetBrains IDEs: chat, task runs,
            background agents, usage, verification and evidence against a
            local faktor-cli daemon over Native Protocol v1.
        """.trimIndent()
        vendor {
            name = "Faktor"
        }
        ideaVersion {
            sinceBuild = "241"
            untilBuild = provider { null }
        }
    }

    pluginVerification {
        ides {
            current()
        }
    }
}

repositories {
    mavenCentral()
    intellijPlatform {
        defaultRepositories()
    }
}

dependencies {
    implementation(project(":backend"))
    implementation(project(":shared"))

    intellijPlatform {
        intellijIdeaCommunity("2024.1.7")
    }
}
