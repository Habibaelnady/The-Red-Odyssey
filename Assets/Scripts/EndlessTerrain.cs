
using System.Collections.Generic;
using UnityEngine;

public class EndlessTerrain : MonoBehaviour
{
    [Header("References")]
    public Transform player;
    public GameObject[] chunkPrefabs;

    [Header("Chunk Settings")]
    public float chunkLength = 200f;
    public int chunksAtStart = 4;
    public int chunksAhead = 4;
    public int chunksBehind = 1;

    private List<GameObject> activeChunks =
        new List<GameObject>();

    private float nextSpawnZ = 0f;

    void Start()
    {
        if (player == null || chunkPrefabs == null ||
            chunkPrefabs.Length == 0)
        {
            Debug.LogError("Missing player or terrain prefabs!");
            enabled = false;
            return;
        }

        // Generate the initial terrain
        for (int i = 0; i < chunksAtStart; i++)
        {
            SpawnNextChunk();
        }
    }

    void Update()
    {
        // Spawn new terrain ahead of the player
        while (nextSpawnZ <
               player.position.z + chunksAhead * chunkLength)
        {
            SpawnNextChunk();
        }

        // Remove old terrain behind the player
        while (activeChunks.Count > 0)
        {
            GameObject oldestChunk = activeChunks[0];

            float oldestEndZ =
                oldestChunk.transform.position.z + chunkLength;

            if (oldestEndZ <
                player.position.z - chunksBehind * chunkLength)
            {
                Destroy(oldestChunk);
                activeChunks.RemoveAt(0);
            }
            else
            {
                break;
            }
        }
    }

    void SpawnNextChunk()
    {
        // Select a random terrain prefab
        GameObject prefab = chunkPrefabs[
            Random.Range(0, chunkPrefabs.Length)
        ];

        // Create the new terrain
        GameObject newChunk = Instantiate(
            prefab,
            new Vector3(0f, 0f, nextSpawnZ),
            Quaternion.identity
        );

        activeChunks.Add(newChunk);

        // Move the spawn position forward
        nextSpawnZ += chunkLength;
    }
}
